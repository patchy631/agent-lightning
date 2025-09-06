#!/usr/bin/env python3
"""
End-to-end Fine-Tuning & Serverless Deployment on Azure ML (Model-as-a-Service)

https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/fine-tune-serverless?tabs=chat-completion&pivots=foundry-portal
https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/finetuning/standalone/model-as-a-service/chat-completion/chat_completion_with_model_as_service.ipynb

This script is a runnable version of the notebook walkthrough. It:
  1) Installs/validates dependencies
  2) Authenticates to Azure using DefaultAzureCredential with Interactive fallback
  3) Connects to your Azure ML Workspace (from config or explicit args)
  4) Registers training & validation data assets (train.jsonl / validation.jsonl)
  5) Creates & submits a fine-tuning job (CHAT_COMPLETION) for the chosen model
  6) Polls for job completion and fetches the registered fine-tuned model name
  7) Creates a Serverless Endpoint using the fine-tuned model (same workspace)
  8) Performs a sample inference call
  9) (Optional) Deletes the serverless endpoint

Usage examples:
  # Use config (azureml folder) and default model
  python azure_finetune_serverless.py --use-config \
      --train-path ./train.jsonl --val-path ./validation.jsonl

  # Provide workspace args explicitly
  python azure_finetune_serverless.py \
      --subscription-id "<SUBSCRIPTION_ID>" \
      --resource-group "<RESOURCE_GROUP_NAME>" \
      --workspace "<WORKSPACE_NAME>" \
      --train-path ./train.jsonl --val-path ./validation.jsonl

Notes:
- Make sure train.jsonl and validation.jsonl are present at the given paths.
- For third-party marketplace models, the script attempts a marketplace subscription.
  This is skipped/ignored for Microsoft 1P models like the Phi family.
"""

import argparse
import json
import os
import sys
import time
import uuid
import subprocess
from typing import Optional
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.ai.ml.finetuning import FineTuningTaskType, create_finetuning_job
from azure.ai.ml.entities import MarketplaceSubscription, ServerlessEndpoint


def get_credential():
    """
    Try DefaultAzureCredential, fall back to InteractiveBrowserCredential.
    """
    try:
        cred = DefaultAzureCredential()
        # Check we can get a token
        cred.get_token("https://management.azure.com/.default")
        print("[auth] Using DefaultAzureCredential")
        return cred
    except Exception as ex:
        print(f"[auth] DefaultAzureCredential failed: {ex}")
        print("[auth] Falling back to InteractiveBrowserCredential...")
        return InteractiveBrowserCredential()


def get_workspace_ml_client(
    cred, use_config: bool, subscription_id: Optional[str], resource_group: Optional[str], workspace: Optional[str]
) -> MLClient:
    """
    Create an MLClient either from local config or explicit parameters.
    """
    if use_config:
        print("[aml] Connecting from local config...")
        try:
            return MLClient.from_config(credential=cred)
        except Exception as ex:
            print(f"[aml] MLClient.from_config failed: {ex}")
            print("[aml] Provide explicit --subscription-id, --resource-group, --workspace")
            raise
    else:
        if not all([subscription_id, resource_group, workspace]):
            raise ValueError(
                "When not using --use-config, you must provide --subscription-id, --resource-group, and --workspace"
            )
        print(f"[aml] Connecting to workspace: sub={subscription_id} rg={resource_group} ws={workspace}")
        return MLClient(
            cred, subscription_id=subscription_id, resource_group_name=resource_group, workspace_name=workspace
        )


def get_or_create_uri_file_asset(ml_client: MLClient, local_path: str, name: str, version: str) -> Data:
    """
    Get or create a URI_FILE data asset pointing to local_path.
    """
    try:
        asset = ml_client.data.get(name, version=version)
        print(f"[data] Dataset '{name}:{version}' already exists.")
        return asset
    except Exception:
        print(f"[data] Creating dataset '{name}:{version}' from path: {local_path}")
        data = Data(
            path=local_path,
            type=AssetTypes.URI_FILE,
            description=f"Dataset for {name}",
            name=name,
            version=version,
        )
        return ml_client.data.create_or_update(data)


def submit_and_wait_finetune_job(
    ml_client: MLClient,
    model_id: str,
    train_data_id: str,
    val_data_id: Optional[str],
    model_name_prefix: str,
    job_display_name: str,
    job_name: str,
    experiment_name: str,
):
    """
    Create a fine-tuning job and poll until it reaches a terminal state.
    """
    print("[ft] Creating fine-tuning job...")
    finetuning_job = create_finetuning_job(
        task=FineTuningTaskType.CHAT_COMPLETION,
        training_data=train_data_id,
        validation_data=val_data_id,
        hyperparameters={
            "per_device_train_batch_size": "1",
            "learning_rate": "0.00002",
            "num_train_epochs": "1",
        },
        model=model_id,
        display_name=job_display_name,
        name=job_name,
        experiment_name=experiment_name,
        tags={"example": "maas-ft"},
        properties={"created_by": "azure_finetune_serverless.py"},
        output_model_name_prefix=model_name_prefix,
    )

    created_job = ml_client.jobs.create_or_update(finetuning_job)
    print(f"[ft] Submitted job: {created_job.name} | status={created_job.status}")

    # Poll for completion
    terminal = {"Completed", "Failed", "Canceled"}
    while True:
        job = ml_client.jobs.get(created_job.name)
        print(f"[ft] Current job status: {job.status}")
        if job.status in terminal:
            print(f"[ft] Job finished with status: {job.status}")
            return job
        time.sleep(30)


def maybe_create_marketplace_subscription(ml_client: MLClient, base_model_id: str, normalized_model_name: str):
    """
    Attempt a marketplace subscription for third-party models.
    Skip for Microsoft 1P models (e.g., Phi family).
    If not required or already exists, this will be ignored.
    """
    try:
        model_id_to_subscribe = "/".join(base_model_id.split("/")[:-2])
        subscription_name = f"{normalized_model_name}-sub"
        ms = MarketplaceSubscription(model_id=model_id_to_subscribe, name=subscription_name)
        print(f"[marketplace] Creating/ensuring subscription: {subscription_name}")
        ml_client.marketplace_subscriptions.begin_create_or_update(ms).result()
        print("[marketplace] Subscription created/verified.")
    except Exception as ex:
        print(f"[marketplace] Skipping or already subscribed: {ex}")


def create_serverless_endpoint(ml_client: MLClient, endpoint_name: str, model_id: str):
    """
    Create or update a ServerlessEndpoint for the fine-tuned model.
    """
    print(f"[endpoint] Creating/Updating serverless endpoint: {endpoint_name}")
    se = ServerlessEndpoint(name=endpoint_name, model_id=model_id)
    ml_client.serverless_endpoints.begin_create_or_update(se).result()
    print("[endpoint] Endpoint is ready.")


def run_sample_inference(ml_client: MLClient, endpoint_name: str, user_message: str):
    """
    Run a basic chat-completions request against the serverless endpoint.
    """
    from urllib.parse import urljoin
    import requests

    endpoint = ml_client.serverless_endpoints.get(endpoint_name)
    keys = ml_client.serverless_endpoints.get_keys(endpoint_name)
    auth_key = keys.primary_key

    url = f"{endpoint.scoring_uri}/v1/chat/completions"
    payload = {
        "max_tokens": 256,
        "messages": [{"role": "user", "content": user_message}],
    }
    headers = {"Content-Type": "application/json", "Authorization": f"{auth_key}"}

    print(f"[infer] POST {url}")
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    print("[infer] Response JSON:")
    print(json.dumps(r.json(), indent=2))


def maybe_delete_endpoint(ml_client: MLClient, endpoint_name: str, do_delete: bool):
    if do_delete:
        print(f"[cleanup] Deleting endpoint: {endpoint_name}")
        ml_client.serverless_endpoints.begin_delete(endpoint_name).result()
        print("[cleanup] Endpoint deleted.")


def main():
    parser = argparse.ArgumentParser(description="Azure ML Fine-tune & Serverless Deploy (MaaS)")
    parser.add_argument("--use-config", action="store_true", help="Use MLClient.from_config (azureml config)")
    parser.add_argument("--subscription-id", type=str, default=None, help="Azure subscription ID")
    parser.add_argument("--resource-group", type=str, default=None, help="Azure resource group")
    parser.add_argument("--workspace", type=str, default=None, help="Azure ML workspace name")

    parser.add_argument(
        "--model-name",
        type=str,
        default="Phi-4-mini-instruct",
        help="Base model name in system registry (e.g., Phi-4-mini-instruct)",
    )
    parser.add_argument("--train-path", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--val-path", type=str, required=True, help="Path to validation.jsonl")

    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=None,
        help="Optional serverless endpoint name (must be unique in workspace)",
    )
    parser.add_argument("--skip-infer", action="store_true", help="Skip the sample inference call")
    parser.add_argument("--delete-endpoint", action="store_true", help="Delete the endpoint at the end")

    args = parser.parse_args()

    # Auth
    cred = get_credential()

    # Workspace client
    ml_client = get_workspace_ml_client(
        cred,
        use_config=args.use_config,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace=args.workspace,
    )

    # Also need a registry client for system registry "azureml"
    registry_ml_client = MLClient(cred, registry_name="azureml")

    # Workspace details
    workspace = ml_client._workspaces.get(ml_client.workspace_name)
    print(f"[aml] Workspace: name={ml_client.workspace_name} location={workspace.location}")
    # Try both attributes for workspace ID (SDK versions differ)
    ws_guid = getattr(workspace, "_workspace_id", None)
    if not ws_guid:
        # For newer SDKs, workspace.id is full ARM ID; we only need the workspace name for AzureML URI.
        # However, azureml:// URIs typically use the GUID. We attempt to parse if possible.
        # If not available, fall back to the name (works in many cases).
        ws_guid = getattr(workspace, "workspace_id", None) or ml_client.workspace_name

    # Pick a model to fine-tune
    model_name = args.model_name
    try:
        model_to_finetune = registry_ml_client.models.get(model_name, label="latest")
    except Exception as ex:
        print(f"[model] Failed to get model: {model_name}")
        available_models = [m.name for m in registry_ml_client.models.list()]
        print(f"[model] Available models: {available_models}")
        raise
    print(f"[model] Using: name={model_to_finetune.name} version={model_to_finetune.version}")
    base_model_id = model_to_finetune.id
    normalized_model_name = model_name.replace(".", "-")

    # Optionally create marketplace subscription for 3P models (ignored for 1P like Phi family)
    maybe_create_marketplace_subscription(ml_client, base_model_id, normalized_model_name)

    # Create training/validation data assets
    # Use version suffix to avoid collisions
    # version_suffix = time.strftime("%Y%m%d%H%M%S")
    version_name = "1"
    train_asset = get_or_create_uri_file_asset(ml_client, args.train_path, "chat_training_small", version_name)
    val_asset = get_or_create_uri_file_asset(ml_client, args.val_path, "chat_validation_small", version_name)

    # Create fine-tune job & wait
    guid = str(uuid.uuid4())[:8]
    display_name = f"{model_name}-display-name-{guid}-from-sdk"
    job_name = f"{model_name}-ft-{guid}-from-sdk"
    out_prefix = f"{model_name}-{guid}-from-sdk-finetuned"
    experiment = f"{model_name}-from-sdk"

    job = submit_and_wait_finetune_job(
        ml_client=ml_client,
        model_id=base_model_id,
        train_data_id=train_asset.id,
        val_data_id=val_asset.id,
        model_name_prefix=out_prefix,
        job_display_name=display_name,
        job_name=job_name,
        experiment_name=experiment,
    )

    # Fetch registered model name from job outputs
    try:
        finetune_model_name = job.outputs["registered_model"]["name"]
        print(f"[ft] Registered fine-tuned model name: {finetune_model_name}")
    except Exception as ex:
        print(f"[ft] Could not read registered model name from job outputs: {ex}")
        print("[ft] Exiting early.")
        sys.exit(2)

    # Build AzureML model URI for serverless endpoint
    model_id_uri = (
        f"azureml://locations/{workspace.location}/workspaces/{ws_guid}/models/{finetune_model_name}/versions/1"
    )
    print(f"[endpoint] Using model id URI: {model_id_uri}")

    # Create endpoint name (must be unique)
    endpoint_name = args.endpoint_name or f"{normalized_model_name}-ft-{guid}"
    create_serverless_endpoint(ml_client, endpoint_name, model_id_uri)

    # Sample inference
    if not args.skip_infer:
        try:
            run_sample_inference(
                ml_client,
                endpoint_name,
                user_message="Summarize the following dialogue in 3 sentences.\nAmanda: I loved the film!\nThierry: Same here, the pacing was perfect.\nAmanda: And the soundtrack? Phenomenal.",
            )
        except Exception as ex:
            print(f"[infer] Inference failed: {ex}")

    # Optional cleanup
    maybe_delete_endpoint(ml_client, endpoint_name, args.delete_endpoint)


if __name__ == "__main__":
    main()
