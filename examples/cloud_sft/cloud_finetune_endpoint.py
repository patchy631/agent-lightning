import json
import logging
import os
import time
import tempfile
import subprocess
import requests
from typing import Union, List, Optional, Any, cast
from openai import OpenAI

from agentlightning.types import LLM, Rollout, TaskInput, Task, NamedResources, ResourcesUpdate
from agentlightning.client import DevTaskLoader

logger = logging.getLogger("agentlightning")


class AzureOpenAIFinetuneEndpoint(DevTaskLoader):
    """
    A DevTaskLoader extension that performs periodic fine-tuning on Azure OpenAI.

    This class collects rollouts and triggers fine-tuning every N tasks, updating
    the LLM endpoint to use the newly fine-tuned model.

    The class currently operates in a single-process single-thread mode.
    """

    def __init__(
        self,
        tasks: Union[List[TaskInput], List[Task]],
        base_deployment_name: str,
        deployment_name: str,
        finetune_every_n_tasks: int = 10,
        azure_openai_endpoint: Optional[str] = None,
        azure_openai_api_key: Optional[str] = None,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        resource_name: Optional[str] = None,
        seed: int = 42,
        n_epochs: int = 3,
        **kwargs,
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
            n_epochs: Number of epochs for fine-tuning
            **kwargs: Additional arguments for DevTaskLoader
        """
        # Initialize base resources with initial model
        self.azure_openai_endpoint = cast(str, azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"))
        if not self.azure_openai_endpoint:
            raise ValueError("Azure OpenAI endpoint must be provided via parameter or AZURE_OPENAI_ENDPOINT env var")

        initial_resources: NamedResources = {
            "main_llm": LLM(endpoint=self.azure_openai_endpoint, model=base_deployment_name)
        }

        super().__init__(tasks=tasks, resources=initial_resources, **kwargs)

        self.base_deployment_name = base_deployment_name
        self.deployment_name = deployment_name
        self.finetune_every_n_tasks = finetune_every_n_tasks

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
        self.n_epochs = n_epochs

        # Tracking
        self.completed_rollouts = []
        self.finetune_count = 0

        # OpenAI client
        if self.azure_openai_endpoint and self.azure_openai_api_key:
            self.openai_client = OpenAI(
                api_key=self.azure_openai_api_key,
                base_url=self.azure_openai_endpoint,
            )
        else:
            self.openai_client = None
            logger.warning("OpenAI client not initialized. Fine-tuning will be skipped.")

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
                self.current_model = fine_tuned_model

                # Deploy the model if Azure parameters are configured
                if all([self.subscription_id, self.resource_group, self.resource_name]):
                    self._deploy_model(fine_tuned_model)

                    # Return updated LLM configuration
                    return LLM(endpoint=self.azure_openai_endpoint or "", model=self.deployment_name)
                else:
                    logger.info("Deployment skipped - Azure parameters not configured")
                    return LLM(endpoint=self.azure_openai_endpoint or "", model=fine_tuned_model)

        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")

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
            # Create a chat-format training example
            messages = []

            # The rollout contains input and output from task execution
            # We'll create training data from the task/response pairs

            # Add a generic system message if needed
            messages.append({"role": "system", "content": "You are a helpful assistant."})

            # Add user message - use rollout_id as a simple prompt
            # In practice, you'd extract actual task details from your specific task format
            messages.append({"role": "user", "content": f"Task {rollout.rollout_id}"})

            # Add assistant response - use a simple acknowledgment
            # In practice, you'd extract actual response from rollout data
            messages.append({"role": "assistant", "content": f"Completed task {rollout.rollout_id} successfully."})

            training_data.append({"messages": messages})

        import pdb

        pdb.set_trace()

        return training_data

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

    def _deploy_model(self, model_name: str) -> None:
        """
        Deploy the fine-tuned model using Azure control plane.

        Args:
            model_name: The fine-tuned model name
        """
        try:
            # Get Azure token
            token = self._get_azure_token()
            if not token:
                logger.error("Could not obtain Azure token for deployment")
                return

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
                        "version": "1",
                    }
                },
            }

            logger.info(f"Deploying model to {self.deployment_name}...")

            response = requests.put(
                request_url,
                params={"api-version": "2024-10-01"},
                headers=headers,
                data=json.dumps(deploy_data),
                timeout=180,
            )

            if response.status_code < 400:
                logger.info(f"Deployment successful: {self.deployment_name}")
            else:
                logger.error(f"Deployment failed: {response.status_code} {response.text}")

        except Exception as e:
            logger.error(f"Error during deployment: {e}")

    def _get_azure_token(self) -> Optional[str]:
        """
        Get Azure management token using Azure CLI.

        Returns:
            Bearer token for Azure management API
        """
        try:
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
            token = subprocess.check_output(cmd, text=True).strip()
            if token:
                return token
        except Exception as e:
            logger.debug(f"Could not fetch token from Azure CLI: {e}")
        return None
