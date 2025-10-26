from twenty_questions import main

main(
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
    output_file="test-results/twenty_questions_qwen30b_withtool41_gpt5mini_20251027.jsonl",
    port=4005,
    search_tool=True,
)
