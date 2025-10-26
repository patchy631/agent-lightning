from twenty_questions import main

main(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    output_file="test-results/twenty_questions_qwen4b_withtool41_gpt5mini_20251027.jsonl",
    port=4004,
    search_tool=True,
)
