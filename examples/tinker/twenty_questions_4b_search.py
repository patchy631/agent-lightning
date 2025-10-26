from twenty_questions import main

main(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    output_file="test-results/twenty_questions_qwen4b_withtool_gpt5mini_20251026.jsonl",
    port=4004,
    search_tool=True,
)
