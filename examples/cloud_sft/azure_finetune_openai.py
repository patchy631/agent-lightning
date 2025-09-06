#!/usr/bin/env python3
"""
azure_finetune_openai.py
End-to-end Fine-Tuning & Deployment on Azure OpenAI (via OpenAI Python SDK + Azure Control Plane)

Docs the script is based on (from your tutorial excerpt):
- Fine-tuning overview and SDK usage (Azure OpenAI in Azure AI Foundry Models)
- LoRA-based fine-tuning, JSONL training format, checkpoints, results.csv
- Control plane deployment (PUT to Azure Management API using 2024-10-01)

What this script does:
  1) Validates dependencies & environment (AZURE_OPENAI_API_KEY + resource base_url)
  2) Creates an OpenAI client pointing at your Azure OpenAI resource
  3) Uploads train/validation JSONL files (UTF-8 with BOM, < 512 MB)
  4) Starts a fine-tuning job (with optional seed / hyperparameters / suffix)
  5) Polls job status to completion (prints events along the way if requested)
  6) Lists checkpoints and (optionally) downloads results.csv
  7) Deploys the fine-tuned model via Azure CONTROL PLANE (management.azure.com)
  8) Runs a sample chat completion against the deployed model (data plane)
  9) (Optional) Deletes the deployment

Usage examples:

  # Minimal (uses environment AZURE_OPENAI_API_KEY and base_url):
  python azure_finetune_openai.py \
      --resource-base-url "https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/" \
      --train-path ./training_set.jsonl \
      --val-path ./validation_set.jsonl

  # Specify model, seed, and a suffix to tag your FT model:
  python azure_finetune_openai.py \
      --resource-base-url "https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/" \
      --model "gpt-4.1-2025-04-14" \
      --seed 105 \
      --suffix "trialA" \
      --train-path ./training_set.jsonl \
      --val-path ./validation_set.jsonl

  # Deploy with control plane and test inference:
  python azure_finetune_openai.py \
      --resource-base-url "https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/" \
      --train-path ./training_set.jsonl \
      --val-path ./validation_set.jsonl \
      --do-deploy \
      --subscription-id "<SUBSCRIPTION_ID>" \
      --resource-group "<RESOURCE_GROUP_NAME>" \
      --resource-name "<YOUR_AZURE_OPENAI_RESOURCE_NAME>" \
      --deployment-name "gpt41-ft-demo" \
      --test-prompt "Tell me a haiku about fine-tuning."

  # Continue training from a previously fine-tuned model:
  python azure_finetune_openai.py \
      --resource-base-url "https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/" \
      --model "gpt-4.1-2025-04-14.ft-5fd1918ee65d4cd38a5dcf6835066ed7" \
      --train-path ./training_set_v2.jsonl \
      --val-path ./validation_set_v2.jsonl

Notes:
- Ensure your JSONL uses the Chat Completions conversational format.
- You need at least 10 training examples; hundreds+ recommended.
- Fine-tuning access requires the appropriate Azure role (e.g., Cognitive Services OpenAI Contributor).
- Deployment uses the Azure CONTROL PLANE (ARM) and requires an access token:
    - Provide via --token, or
    - Have Azure CLI installed and logged in; script will attempt to call:
      az account get-access-token --resource https://management.azure.com
"""

import argparse
import json
import os
import sys
import time
import subprocess
from typing import Optional, Dict, Any

# OpenAI Python SDK (1.x)
try:
    from openai import OpenAI
except Exception as ex:
    print("[setup] Missing dependency: openai (pip install openai)")
    raise

import requests


# ---------- Helpers ----------


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def require_file(path: str, label: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found at: {path}")
    size = os.path.getsize(path)
    if size >= 512 * 1024 * 1024:
        raise ValueError(f"{label} must be < 512 MB; got {size} bytes")
    return path


def get_openai_client(resource_base_url: str, api_key: Optional[str]) -> OpenAI:
    """
    Create an OpenAI client pointed to the Azure OpenAI *data plane* for your resource.
    base_url example: https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/
    """
    if not resource_base_url:
        raise ValueError("--resource-base-url is required (your Azure OpenAI data-plane endpoint with /openai/v1/)")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY must be set in env or passed via --api-key")

    client = OpenAI(
        api_key=api_key,
        base_url=resource_base_url,
    )
    return client


def upload_files(client: OpenAI, train_path: str, val_path: Optional[str]):
    """
    Upload training and validation files. Returns (train_file_id, val_file_id or None)
    """
    print("[files] Uploading training file...")
    training_response = client.files.create(file=open(train_path, "rb"), purpose="fine-tune")
    train_id = training_response.id
    print(f"[files] Training file ID: {train_id}")

    val_id = None
    if val_path:
        print("[files] Uploading validation file...")
        validation_response = client.files.create(file=open(val_path, "rb"), purpose="fine-tune")
        val_id = validation_response.id
        print(f"[files] Validation file ID: {val_id}")

    print("[files] waiting 10 seconds for files to be processed...")
    time.sleep(10)  # give the service a moment to process the files
    return train_id, val_id


def create_finetune_job(
    client: OpenAI,
    model: str,
    training_file_id: str,
    validation_file_id: Optional[str],
    seed: Optional[int],
    hyperparams: Optional[Dict[str, Any]],
    suffix: Optional[str],
) -> str:
    """
    Starts a fine-tuning job. Returns the job_id.
    """
    payload: Dict[str, Any] = {
        "training_file": training_file_id,
        "model": model,
    }
    if validation_file_id:
        payload["validation_file"] = validation_file_id
    if seed is not None:
        payload["seed"] = int(seed)
    if hyperparams:
        payload["hyperparameters"] = hyperparams
    if suffix:
        payload["suffix"] = suffix  # helps distinguish iterations

    print("[ft] Creating fine-tuning job...")
    print(f"[ft] Payload: {json.dumps(payload, indent=2)}")
    resp = client.fine_tuning.jobs.create(**payload)
    job_id = resp.id
    print(f"[ft] Job created: {job_id}")
    return job_id


def poll_job(client: OpenAI, job_id: str, show_events: bool = False, interval_sec: int = 15) -> Dict[str, Any]:
    """
    Polls until the job reaches a terminal state. Returns the final job object (dict-like).
    """
    terminal = {"succeeded", "failed", "cancelled"}
    last_status = None
    printed_event_ids = set()

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        if status != last_status:
            print(f"[ft] Status: {status}")
            last_status = status

        if show_events:
            try:
                events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=20)
                for ev in getattr(events, "data", []):
                    if ev.id not in printed_event_ids:
                        printed_event_ids.add(ev.id)
                        ts = getattr(ev, "created_at", None)
                        msg = getattr(ev, "message", "")
                        print(f"[ft][event] {ts}: {msg}")
            except Exception as ex:
                eprint(f"[ft][event] Could not list events yet: {ex}")

        if status in terminal:
            print(f"[ft] Job finished with status: {status}")
            # Convert to plain dict for consistent downstream handling
            return json.loads(job.model_dump_json())

        time.sleep(interval_sec)


def list_checkpoints(client: OpenAI, job_id: str):
    print("[ft] Listing checkpoints...")
    try:
        resp = client.fine_tuning.jobs.checkpoints.list(job_id)
        obj = json.loads(resp.model_dump_json())
        checkpoints = obj.get("data", [])
        for i, ck in enumerate(checkpoints, 1):
            print(f"  [{i}] id={ck.get('id')}  created_at={ck.get('created_at')}  step={ck.get('step')}")
        return checkpoints
    except Exception as ex:
        eprint(f"[ft] Could not list checkpoints: {ex}")
        return []


def maybe_download_results_csv(client: OpenAI, job_final: Dict[str, Any], out_dir: str):
    """
    If the job succeeded and a result file is present, download it.
    """
    status = job_final.get("status")
    if status != "succeeded":
        print("[results] Job not successful; skipping results.csv download.")
        return None

    # The tutorial indicates job.result_files[0] will exist when succeeded
    result_files = job_final.get("result_files") or []
    if not result_files:
        print("[results] No result_files found in job; skipping.")
        return None

    file_id = result_files[0]
    print(f"[results] Downloading results file: {file_id}")

    retrieve = client.files.retrieve(file_id)
    filename = getattr(retrieve, "filename", None) or "results.csv"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "wb") as f:
        result = client.files.content(file_id).read()
        f.write(result)

    print(f"[results] Saved to: {out_path}")
    return out_path


def get_token_from_cli() -> Optional[str]:
    """
    Attempt to fetch an ARM token via Azure CLI.
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
            print("[auth] Obtained ARM token from Azure CLI.")
            return token
    except Exception as ex:
        eprint(f"[auth] Could not fetch token from Azure CLI: {ex}")
    return None


def deploy_control_plane(
    token: str,
    subscription_id: str,
    resource_group: str,
    resource_name: str,
    model_name: str,
    deployment_name: str,
    api_version: str = "2024-10-01",
    sku_name: str = "standard",
    capacity: int = 1,
) -> Dict[str, Any]:
    """
    Creates/updates a deployment for the fine-tuned model using Azure CONTROL PLANE.
    model_name example: gpt-4.1-2025-04-14.ft-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    """
    if not all([token, subscription_id, resource_group, resource_name, model_name, deployment_name]):
        raise ValueError("Missing required parameters for control-plane deployment")

    request_url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.CognitiveServices/accounts/{resource_name}"
        f"/deployments/{deployment_name}"
    )

    deploy_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    deploy_data = {
        "sku": {"name": sku_name, "capacity": capacity},
        "properties": {
            "model": {
                "format": "OpenAI",
                "name": model_name,
                "version": "1",
            }
        },
    }

    print(f"[deploy] PUT {request_url}?api-version={api_version}")
    r = requests.put(
        request_url,
        params={"api-version": api_version},
        headers=deploy_headers,
        data=json.dumps(deploy_data),
        timeout=180,
    )
    try:
        payload = r.json()
    except Exception:
        payload = {"text": r.text}

    print(f"[deploy] HTTP {r.status_code} {r.reason}")
    print(json.dumps(payload, indent=2))
    if r.status_code >= 400:
        raise RuntimeError(f"Deployment failed: HTTP {r.status_code}")
    return payload


def delete_deployment_control_plane(
    token: str,
    subscription_id: str,
    resource_group: str,
    resource_name: str,
    deployment_name: str,
    api_version: str = "2024-10-01",
):
    request_url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.CognitiveServices/accounts/{resource_name}"
        f"/deployments/{deployment_name}"
    )
    headers = {
        "Authorization": f"Bearer {token}",
    }
    print(f"[cleanup] DELETE {request_url}?api-version={api_version}")
    r = requests.delete(
        request_url,
        params={"api-version": api_version},
        headers=headers,
        timeout=180,
    )
    print(f"[cleanup] HTTP {r.status_code} {r.reason}")
    if r.status_code >= 400:
        eprint(f"[cleanup] Delete may have failed: {r.text}")


def run_sample_inference(client: OpenAI, deployment_name: str, user_message: str):
    """
    Data-plane inference using OpenAI SDK against Azure OpenAI.
    In Azure OpenAI with the OpenAI SDK, pass model=<deployment_name>.
    """
    print("[infer] Running sample chat.completions...")
    resp = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=256,
    )
    print(json.dumps(json.loads(resp.model_dump_json()), indent=2))


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser(description="Azure OpenAI Fine-Tune & Deploy")
    # Data-plane
    parser.add_argument(
        "--resource-base-url",
        type=str,
        required=True,
        help="Azure OpenAI data-plane base URL (e.g., https://<resource>.openai.azure.com/openai/v1/)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("AZURE_OPENAI_API_KEY"),
        help="Azure OpenAI API key (defaults to env AZURE_OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-2025-04-14",
        help="Base model or an existing fine-tuned model id to continue from.",
    )
    parser.add_argument("--train-path", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--val-path", type=str, default=None, help="Path to validation JSONL (optional)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument("--suffix", type=str, default=None, help="Up to 18 chars to label the fine-tuned model")
    parser.add_argument("--n-epochs", type=int, default=None, help="Optional hyperparameter: n_epochs")
    parser.add_argument(
        "--learning-rate-multiplier", type=float, default=None, help="Optional hyperparameter: learning_rate_multiplier"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional hyperparameter: batch_size")
    parser.add_argument("--show-events", action="store_true", help="Print fine-tuning events while polling")
    parser.add_argument("--download-results", action="store_true", help="Download results.csv on success")
    parser.add_argument("--results-dir", type=str, default="./ft_results", help="Where to save results.csv")

    # Control-plane deploy
    parser.add_argument("--do-deploy", action="store_true", help="Create/Update a deployment via Azure control plane")
    parser.add_argument("--subscription-id", type=str, default=None, help="Azure subscription ID")
    parser.add_argument("--resource-group", type=str, default=None, help="Azure resource group")
    parser.add_argument("--resource-name", type=str, default=None, help="Azure OpenAI resource name")
    parser.add_argument(
        "--deployment-name", type=str, default="gpt41-ft", help="Deployment name to create or update in Azure OpenAI"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("TOKEN"),
        help="ARM bearer token for control-plane calls (tries Azure CLI if not provided)",
    )

    # Inference + cleanup
    parser.add_argument("--test-prompt", type=str, default=None, help="Send a test prompt to the deployed model")
    parser.add_argument("--delete-deployment", action="store_true", help="Delete deployment at the end")

    args = parser.parse_args()

    # Validate files
    train_path = require_file(args.train_path, "Training file")
    val_path = require_file(args.val_path, "Validation file") if args.val_path else None

    # Build hyperparameters payload if provided
    hyperparams = {}
    if args.n_epochs is not None:
        hyperparams["n_epochs"] = int(args.n_epochs)
    if args.learning_rate_multiplier is not None:
        hyperparams["learning_rate_multiplier"] = float(args.learning_rate_multiplier)
    if args.batch_size is not None:
        hyperparams["batch_size"] = int(args.batch_size)
    if not hyperparams:
        hyperparams = None  # let service choose defaults

    # Client
    client = get_openai_client(args.resource_base_url, args.api_key)

    # Upload
    train_id, val_id = upload_files(client, train_path, val_path)

    # Create FT job
    job_id = create_finetune_job(
        client=client,
        model=args.model,
        training_file_id=train_id,
        validation_file_id=val_id,
        seed=args.seed,
        hyperparams=hyperparams,
        suffix=args.suffix,
    )

    # Poll
    job_final = poll_job(client, job_id, show_events=args.show_events, interval_sec=20)

    # Print final job object
    print("[ft] Final job object:")
    print(json.dumps(job_final, indent=2))

    # List checkpoints
    list_checkpoints(client, job_id)

    # Optionally download results.csv
    result_csv_path = None
    if args.download_results:
        result_csv_path = maybe_download_results_csv(client, job_final, args.results_dir)

    # Extract fine-tuned model name/id
    fine_tuned_model = job_final.get("fine_tuned_model")
    if not fine_tuned_model:
        print("[ft] No 'fine_tuned_model' in final job object (job may have failed). Exiting.")
        if job_final.get("status") != "succeeded":
            sys.exit(2)

    print(f"[ft] Fine-tuned model: {fine_tuned_model}")

    # Control-plane deploy (optional)
    if args.do_deploy:
        token = args.token or get_token_from_cli()
        if not token:
            raise RuntimeError("No control-plane token found. Supply --token or login with Azure CLI.")

        if not all([args.subscription_id, args.resource_group, args.resource_name, args.deployment_name]):
            raise ValueError(
                "--subscription-id, --resource-group, --resource-name, and --deployment-name are required with --do-deploy"
            )

        deploy_control_plane(
            token=token,
            subscription_id=args.subscription_id,
            resource_group=args.resource_group,
            resource_name=args.resource_name,
            model_name=fine_tuned_model,
            deployment_name=args.deployment_name,
        )

        # Sample inference (data plane) using the deployment name
        if args.test_prompt:
            try:
                run_sample_inference(client, args.deployment_name, args.test_prompt)
            except Exception as ex:
                eprint(f"[infer] Inference failed: {ex}")

        # Optional cleanup
        if args.delete_deployment:
            delete_deployment_control_plane(
                token=token,
                subscription_id=args.subscription_id,
                resource_group=args.resource_group,
                resource_name=args.resource_name,
                deployment_name=args.deployment_name,
            )

    print("[done] All steps complete.")


if __name__ == "__main__":
    main()
