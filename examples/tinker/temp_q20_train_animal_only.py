# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from q20_train import *


async def algo_animal_only(search: bool, model: Literal["qwen4b", "qwen30b"], port: int):
    raw_data = pd.read_csv("q20_nouns.csv")  # type: ignore
    raw_data["search_enabled"] = search
    train_data, test_data = raw_data[raw_data["split"] == "train"], raw_data[raw_data["split"] == "test"]  # type: ignore

    train_dataset = cast(agl.Dataset[Q20Task], train_data.to_dict(orient="records"))  # type: ignore
    test_dataset = cast(agl.Dataset[Q20Task], test_data.to_dict(orient="records"))  # type: ignore

    if model == "qwen4b":
        model_name = "Qwen/Qwen3-4B-Instruct-2507"
        renderer_name = "qwen3"
    elif model == "qwen30b":
        model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        renderer_name = "qwen3"
    else:
        raise ValueError(f"Invalid model: {model}")

    experiment_name = f"q20_{'search' if search else 'no_search'}_{model}_lr1e-6_animalonly"

    llm_proxy_port = _find_available_port()

    config = Config(
        learning_rate=1e-6,
        dataset_builder=AGLDatasetBuilder(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=8,
            shuffle=True,
            group_size=4,
            seed=42,
            n_epochs=10,
        ),
        renderer_name=renderer_name,
        model_name=model_name,
        log_path=f"logs/{experiment_name}",
        concurrency=16,
        eval_every=4,
        wandb_project="AgentLightningQ20",
        wandb_name=experiment_name,
        store_address=f"http://localhost:{port}",
        llm_proxy_port=llm_proxy_port,
        adapter_from_llm_proxy=False,
        llm_proxy_retry_attempts=5,
    )
    await entrypoint(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Q20 AgentLightning experiments.")
    parser.add_argument("--port", type=int, default=4747, help="Port for the AgentLightning store.")
    parser.add_argument("--search", action="store_true", help="Enable search tool.")
    parser.add_argument(
        "--model",
        choices=("qwen4b", "qwen30b"),
        default="qwen30b",
        help="Model variant to train.",
    )
    args = parser.parse_args()
    agl.configure_logger()
    asyncio.run(algo_animal_only(search=args.search, model=args.model, port=args.port))


if __name__ == "__main__":
    main()
