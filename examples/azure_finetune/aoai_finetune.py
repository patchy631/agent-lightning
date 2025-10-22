import json
import logging
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import requests
from openai import AsyncOpenAI
from rich.console import Console

from agentlightning.adapter.messages import OpenAIMessages
from agentlightning.algorithm import Algorithm
from agentlightning.algorithm.utils import batch_iter_over_dataset
from agentlightning.types import LLM, NamedResources, ResourcesUpdate, Rollout, RolloutMode, TaskInput

console = Console()


class AzureOpenAIFinetune(Algorithm):
    """
    A DevTaskLoader extension that performs periodic fine-tuning on Azure OpenAI.

    This class collects rollouts and triggers fine-tuning every N tasks, updating
    the LLM endpoint to use the newly fine-tuned model.

    The class currently operates in a single-process single-thread mode.
    """

    def __init__(
        self,
        base_deployment_name: str,
        deployment_name: str,
        *,
        finetune_every_n_rollouts: int = 10,
        azure_openai_endpoint: Optional[str] = None,
        azure_openai_api_key: Optional[str] = None,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        resource_name: Optional[str] = None,
        seed: int = 42,
        n_iterations: int = 3,
        data_filter_ratio: float = 0.5,
    ):
        """
        Initialize the Azure OpenAI Fine-tune Endpoint.

        Args:
            tasks: List of tasks to process
            base_deployment_name: Name for the model / deployment to start with
            deployment_name: Name for the deployment after fine-tuning
            finetune_every_n_tasks: Number of tasks to complete before triggering fine-tuning
            azure_openai_endpoint: Azure OpenAI endpoint URL (e.g., https://resource.openai.azure.com/openai/v1/)
            azure_openai_api_key: Azure OpenAI API key
            subscription_id: Azure subscription ID for deployment
            resource_group: Azure resource group for deployment
            resource_name: Azure OpenAI resource name
            seed: Random seed for fine-tuning
            n_iterations: Number of iterations for fine-tuning
            data_filter_ratio: Ratio of data to use for fine-tuning (1.0 = all, 0.5 = half, etc.).
                The data with the highest rewards will be selected. Others will be dropped.
            **kwargs: Additional arguments for DevTaskLoader
        """
        # Initialize base resources with initial model
        self.azure_openai_endpoint = cast(str, azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"))
        if not self.azure_openai_endpoint:
            raise ValueError("Azure OpenAI endpoint must be provided via parameter or AZURE_OPENAI_ENDPOINT env var")

        self.base_deployment_name = base_deployment_name
        self.deployment_name = deployment_name
        self.finetune_every_n_rollouts = finetune_every_n_rollouts

        # Azure deployment parameters
        self.azure_openai_api_key = cast(str, azure_openai_api_key or os.getenv("AZURE_OPENAI_API_KEY"))
        self.subscription_id = cast(str, subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID"))
        self.resource_group = cast(str, resource_group or os.getenv("AZURE_RESOURCE_GROUP"))
        self.resource_name = cast(str, resource_name or os.getenv("AZURE_RESOURCE_NAME"))

        if not self.azure_openai_endpoint:
            raise ValueError("Azure OpenAI endpoint must be provided via parameter or AZURE_OPENAI_ENDPOINT env var")
        if not self.azure_openai_api_key:
            raise ValueError("Azure OpenAI API key must be provided via parameter or AZURE_OPENAI_API_KEY env var")
        if not self.subscription_id:
            raise ValueError("Azure subscription ID must be provided via parameter or AZURE_SUBSCRIPTION_ID env var")
        if not self.resource_group:
            raise ValueError("Azure resource group must be provided via parameter or AZURE_RESOURCE_GROUP env var")
        if not self.resource_name:
            raise ValueError("Azure resource name must be provided via parameter or AZURE_RESOURCE_NAME env var")

        # Fine-tuning parameters
        self.base_model = base_deployment_name
        self.current_model = self.base_model
        self.seed = seed
        self.n_iterations = n_iterations
        self.data_filter_ratio = data_filter_ratio

        # Tracking
        self.completed_rollouts = []
        self.finetune_count = 0

        # OpenAI client
        if self.azure_openai_endpoint and self.azure_openai_api_key:
            self.openai_client = AsyncOpenAI(
                api_key=self.azure_openai_api_key,
                base_url=self.azure_openai_endpoint,
            )
        else:
            self.openai_client = None
            logger.warning("OpenAI client not initialized. Fine-tuning will be skipped.")

    async def run(
        self,
        train_dataset: Optional[List[TaskInput]] = None,
        val_dataset: Optional[List[TaskInput]] = None,
    ):
        """
        Run the training loop.

        Args:
            train_dataset: Optional training dataset
            val_dataset: Optional validation dataset
        """
        if train_dataset is None or val_dataset is None:
            raise ValueError("Both train_dataset and val_dataset must be provided")

        resources: NamedResources = {"main_llm": LLM(endpoint=self.azure_openai_endpoint, model=self.current_model)}
        store = self.get_store()

        data_iterator = batch_iter_over_dataset(train_dataset, self.finetune_every_n_rollouts)
        for i_iteration in range(self.n_iterations):
            # Fetch the next batch of tasks to process
            tasks = next(data_iterator)
            console.print(
                f"[bold red][Algo {i_iteration + 1} / {self.n_iterations}][/bold red] Starting fine-tuning iteration with {len(tasks)} tasks..."
            )

            # Update the current active LLM deployment address
            await store.add_resources(resources)
            console.print(
                f"[bold red][Algo {i_iteration + 1} / {self.n_iterations}][/bold red] Using model deployment: {self.current_model}"
            )

            # Spawn and wait for the rollouts to complete
            messages_group, reward_group = await self.batch_rollout_and_collect_data(tasks)
            console.print(
                f"[bold red][Algo {i_iteration + 1} / {self.n_iterations}][/bold red] Completed rollouts for {len(tasks)} tasks."
            )

            # Filter the data based on rewards
            training_data = await self.prepare_data_for_training(messages_group, reward_group)
            console.print(
                f"[bold red][Algo {i_iteration + 1} / {self.n_iterations}][/bold red] Prepared {len(training_data)} training examples after filtering."
            )

            # Perform fine-tuning
            console.print(
                f"[bold red][Algo {i_iteration + 1} / {self.n_iterations}][/bold red] Starting fine-tuning..."
            )
            finetuned_model_id = self.finetune(training_data)
            console.print(
                f"[bold red][Algo {i_iteration + 1} / {self.n_iterations}][/bold red] Fine-tuning completed. New model: {finetuned_model_id}"
            )

            # Deploy the fine-tuned model
            console.print(
                f"[bold red][Algo {i_iteration + 1} / {self.n_iterations}][/bold red] Deploying fine-tuned model..."
            )
            new_llm = self.deploy_finetuned_model(finetuned_model_id)
            resources = {"main_llm": new_llm}
            console.print(
                f"[bold red][Algo {i_iteration + 1} / {self.n_iterations}][/bold red] Deployment completed. Updated resources to: {new_llm}"
            )

    async def batch_rollout_and_collect_data(self, tasks: List[TaskInput]) -> List[Tuple[OpenAIMessages, float]]:
        """Perform rollouts for a batch of tasks and collect the resulting messages.

        Args:
            tasks: List of tasks to process

        Returns:
            List of OpenAIMessages collected from the rollouts
        """

    async def rollout_and_collect_data(self, task: TaskInput, mode: RolloutMode) -> Tuple[OpenAIMessages, float]:
        """Perform a rollout for a single task and collect the resulting messages.

        Args:
            task: The task to process

        Returns:
            OpenAIMessages collected from the rollout, and the final reward
        """
        store = self.get_store()
        rollout = await store.enqueue_rollout(...)

    def post_rollout(self, rollout: Rollout) -> Optional[dict[str, Any]]:
        """
        Override post_rollout to track completed tasks and trigger fine-tuning.

        Args:
            rollout: The completed rollout

        Returns:
            Response dictionary
        """
        # Call parent implementation
        result = super().post_rollout(rollout)

        # Track the rollout
        self.completed_rollouts.append(rollout)

        # Check if we should trigger fine-tuning
        if len(self.completed_rollouts) >= self.finetune_every_n_tasks:
            logger.info(f"Triggering fine-tuning after {len(self.completed_rollouts)} tasks...")
            new_llm = self.finetune(self.completed_rollouts)

            # Update resources with new model
            if new_llm and new_llm.endpoint:
                self.finetune_count += 1
                new_resources_id = f"finetune_{self.finetune_count}"
                self._resources_update = ResourcesUpdate(resources_id=new_resources_id, resources={"main_llm": new_llm})
                logger.info(f"Updated resources to use fine-tuned model (resources_id: {new_resources_id})")

                # Clear completed rollouts for next batch
                self.completed_rollouts = []

        return result

    def finetune(self, data: List[Rollout]) -> Optional[LLM]:
        """
        Perform fine-tuning on Azure OpenAI using the collected rollouts.

        Args:
            data: List of completed rollouts to use for fine-tuning

        Returns:
            Updated LLM configuration with the new fine-tuned model endpoint
        """
        if not self.openai_client:
            logger.warning("Skipping fine-tuning - OpenAI client not configured")
            return None

        train_file_path = None
        try:
            # Convert rollouts to JSONL training data
            training_data = self._prepare_training_data(data)

            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                for item in training_data:
                    f.write(json.dumps(item) + "\n")
                train_file_path = f.name

            logger.info(f"Created training file with {len(training_data)} examples")

            # Upload training file
            logger.info("Uploading training file...")
            with open(train_file_path, "rb") as f:
                training_response = self.openai_client.files.create(file=f, purpose="fine-tune")
            train_file_id = training_response.id
            logger.info(f"Training file uploaded: {train_file_id}")

            # Wait for file processing
            logger.info("Waiting for training file to be processed...")
            time.sleep(10)

            # Create fine-tuning job
            logger.info("Starting fine-tuning job...")
            ft_job = self.openai_client.fine_tuning.jobs.create(
                training_file=train_file_id,
                model=self.current_model,
                seed=self.seed,
                hyperparameters={"n_epochs": self.n_epochs},
                suffix=f"auto_{self.finetune_count + 1}",
            )
            job_id = ft_job.id
            logger.info(f"Fine-tuning job created: {job_id}")

            # Poll for completion
            fine_tuned_model = self._wait_for_finetuning(job_id)

            if fine_tuned_model:
                logger.info(f"Fine-tuning completed: {fine_tuned_model}")
                # Update the current model for next round.
                # Use continuous fine-tuning feature here.
                self.current_model = fine_tuned_model

                # Deploy the model if Azure parameters are configured
                if all([self.subscription_id, self.resource_group, self.resource_name]):
                    self._deploy_model(fine_tuned_model, str(self.finetune_count + 1))

                    # Return updated LLM configuration
                    return LLM(endpoint=self.azure_openai_endpoint or "", model=self.deployment_name)
                else:
                    logger.info("Deployment skipped - Azure parameters not configured")
                    return LLM(endpoint=self.azure_openai_endpoint or "", model=fine_tuned_model)

        finally:
            # Clean up temporary file
            try:
                if train_file_path:
                    os.unlink(train_file_path)
            except:
                pass

        return None

    def _prepare_training_data(self, rollouts: List[Rollout]) -> List[dict]:
        """
        Convert rollouts to JSONL training format for Azure OpenAI.

        Args:
            rollouts: List of completed rollouts

        Returns:
            List of training examples in chat format
        """
        training_data = []

        for rollout in rollouts:
            tool_calls = []
            prompt_completions = []

            # Ignore rollouts without trace
            if not rollout.trace:
                continue

            for trace in rollout.trace:
                if "attributes" not in trace:
                    continue

                # Otherwise we strip all the tool calls and prompts and responses
                tool_call = convert_genai_dict(trace["attributes"], "tool")
                if tool_call:
                    tool_calls.append(tool_call)

                prompt = convert_genai_dict(trace["attributes"], "gen_ai.prompt")
                completion = convert_genai_dict(trace["attributes"], "gen_ai.completion")
                request = convert_genai_dict(trace["attributes"], "gen_ai.request")
                response = convert_genai_dict(trace["attributes"], "gen_ai.response")
                if prompt or completion or request or response:
                    prompt_completions.append(
                        {
                            "prompt": prompt,
                            "completion": completion,
                            "request": request,
                            "response": response,
                        }
                    )

            # print(tool_calls, prompt_completions)
            for item in convert_to_json_list(prompt_completions, tool_calls):
                # TODO: we always use final reward here
                # ideally this should be replaced with the credit assignment logic
                training_data.append({**item, "reward": rollout.final_reward})
                logger.info(f"Fine-tuning data: {item}")

        return self._filter_training_data(training_data)

    def _filter_training_data(self, data: List[dict]) -> List[dict]:
        """
        Filter the training data based on rewards.

        Args:
            data: List of training examples with 'reward' field

        Returns:
            Filtered list of training examples without 'reward' field
        """
        if self.data_filter_ratio >= 1.0:
            return data

        # Sort by reward descending
        sorted_data = sorted(data, key=lambda x: x.get("reward", 0), reverse=True)
        n_keep = max(1, int(len(sorted_data) * self.data_filter_ratio))
        filtered = sorted_data[:n_keep]

        logger.info(f"Filtered training data: kept {n_keep} out of {len(data)} examples")

        # Remove reward field for fine-tuning
        for item in filtered:
            if "reward" in item:
                del item["reward"]

        return filtered

    def _wait_for_finetuning(self, job_id: str, interval: int = 20) -> Optional[str]:
        """
        Wait for fine-tuning job to complete.

        Args:
            job_id: The fine-tuning job ID
            interval: Polling interval in seconds

        Returns:
            The fine-tuned model name if successful, None otherwise
        """
        terminal_states = {"succeeded", "failed", "cancelled"}

        while True:
            if self.openai_client:
                job = self.openai_client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                logger.debug(f"Fine-tuning job status: {status}")

                if status in terminal_states:
                    if status == "succeeded":
                        return job.fine_tuned_model
                    else:
                        logger.warning(f"Fine-tuning job ended with status: {status}")
                        return None

                time.sleep(interval)
            else:
                return None

    def _deploy_model(self, model_name: str, version: str) -> None:
        """
        Deploy the fine-tuned model using Azure control plane.

        Args:
            model_name: The fine-tuned model name
        """
        # Get Azure token
        token = self._get_azure_token()

        # Prepare deployment request
        request_url = (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.CognitiveServices/accounts/{self.resource_name}"
            f"/deployments/{self.deployment_name}"
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        deploy_data = {
            "sku": {"name": "standard", "capacity": 1},
            "properties": {
                "model": {
                    "format": "OpenAI",
                    "name": model_name,
                    "version": version,
                }
            },
        }

        logger.info(f"Deploying model to {self.deployment_name}...")

        response = requests.put(
            request_url,
            params={"api-version": "2025-06-01"},
            headers=headers,
            data=json.dumps(deploy_data),
            timeout=180,
        )

        if response.status_code < 400:
            logger.info(f"Deployment successful: {self.deployment_name}")
        else:
            logger.error(f"Deployment failed: {response.status_code} {response.text}")

        # TODO: wait for the deployment to be ready

    def _get_azure_token(self) -> str:
        """
        Get Azure management token using Azure CLI.

        Returns:
            Bearer token for Azure management API
        """
        cmd = [
            "az",
            "account",
            "get-access-token",
            "--resource",
            "https://management.azure.com",
            "--query",
            "accessToken",
            "-o",
            "tsv",
        ]
        try:
            token = subprocess.check_output(cmd, text=True).strip()
        except subprocess.CalledProcessError:
            raise ValueError("Azure CLI command failed. Could not fetch token from Azure CLI.")
        if token:
            return token
        else:
            raise ValueError("Could not fetch token from Azure CLI.")
