from openai import OpenAI
import vllm.entrypoints.openai.serving_completion
from verl.utils import hf_tokenizer

openai = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

tokenizer = hf_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)


# Deconstruct result to get token IDs
def deconstruct_result(result, tokenizer):
    """
    Deconstruct OpenAI result to {"prompt_token_ids": list[int], "response_token_ids": list[list[int]]}
    """
    # Extract prompt token IDs from prompt_logprobs
    prompt_token_ids = []
    if hasattr(result, "prompt_logprobs") and result.prompt_logprobs:
        for logprob_dict in result.prompt_logprobs:
            if logprob_dict is not None:
                if len(logprob_dict) != 1:
                    raise ValueError(f"logprob_dict should have length 1, but got: {logprob_dict}")
                prompt_token = next(iter(logprob_dict.keys()))
                try:
                    token_id = int(prompt_token)
                    prompt_token_ids.append(token_id)
                except ValueError:
                    # If token is a string, convert using tokenizer
                    decoded_token = logprob_dict[best_token].get("decoded_token", best_token)
                    token_ids = tokenizer.encode(decoded_token, add_special_tokens=False)
                    if len(token_ids) != 1:
                        raise ValueError(f"Tokenizer failed to encode token: {decoded_token}")
                    prompt_token_ids.extend(token_ids)

    # Extract response token IDs from choices
    response_token_ids = []
    for choice in result.choices:
        choice_tokens = []
        if choice.logprobs and choice.logprobs.content:
            for token_logprob in choice.logprobs.content:
                token_str = token_logprob.token
                try:
                    # Try to encode the token string to get token ID
                    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                    if len(token_ids) != 1:
                        raise ValueError(f"Tokenizer failed to encode token: {token_str}")
                    choice_tokens.extend(token_ids)
                except Exception as e:
                    raise ValueError(f"Failed to get token ID for '{token_str}': {e}")
        response_token_ids.append(choice_tokens)

    return {"prompt_token_ids": prompt_token_ids, "response_token_ids": response_token_ids}


result = openai.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_capital",
                "description": "Get the capital of a country",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {"type": "string", "description": "The name of the country to get the capital for."}
                    },
                    "required": ["country"],
                },
            },
        }
    ],
    logprobs=True,
    extra_body={"prompt_logprobs": 1, "return_tokens_as_token_ids": True},
)
print("Result:", result.choices[0].message)
print("Result:", result)

# Test the deconstruction
deconstructed = deconstruct_result(result, tokenizer)
print("\nDeconstructed result:")
print(f"Prompt token IDs: {deconstructed['prompt_token_ids']}")
print(f"Response token IDs: {deconstructed['response_token_ids']}")


# Result: ChatCompletion(id='chatcmpl-6ec93f5ae1e54f44a1b936c19ac2c056', choices=[Choice(finish_reason='stop', index=0, logprobs=ChoiceLogprobs(content=[ChatCompletionTokenLogprob(token='The', bytes=[84, 104, 101], logprob=-0.03290422633290291, top_logprobs=[]), ChatCompletionTokenLogprob(token=' capital', bytes=[32, 99, 97, 112, 105, 116, 97, 108], logprob=-0.00022301571152638644, top_logprobs=[]), ChatCompletionTokenLogprob(token=' of', bytes=[32, 111, 102], logprob=-0.030539512634277344, top_logprobs=[]), ChatCompletionTokenLogprob(token=' France', bytes=[32, 70, 114, 97, 110, 99, 101], logprob=-0.00011300401820335537, top_logprobs=[]), ChatCompletionTokenLogprob(token=' is', bytes=[32, 105, 115], logprob=-0.0002571013756096363, top_logprobs=[]), ChatCompletionTokenLogprob(token=' Paris', bytes=[32, 80, 97, 114, 105, 115], logprob=-0.00026663561584427953, top_logprobs=[]), ChatCompletionTokenLogprob(token='.', bytes=[46], logprob=-0.011647789739072323, top_logprobs=[]), ChatCompletionTokenLogprob(token='<|im_end|>', bytes=[60, 124, 105, 109, 95, 101, 110, 100, 124, 62], logprob=-0.0054404293186962605, top_logprobs=[])], refusal=None), message=ChatCompletionMessage(content='The capital of France is Paris.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[], reasoning_content=None), stop_reason=None)], created=1754809371, model='Qwen/Qwen2.5-Coder-3B-Instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=8, prompt_tokens=26, total_tokens=34, completion_tokens_details=None, prompt_tokens_details=None), prompt_logprobs=[None, {'8948': {'logprob': -14.364860534667969, 'rank': 56256, 'decoded_token': 'system'}, '198': {'logprob': -0.5523606538772583, 'rank': 1, 'decoded_token': '\n'}}, {'198': {'logprob': -0.8552187085151672, 'rank': 1, 'decoded_token': '\n'}}, {'2610': {'logprob': -9.523348808288574, 'rank': 496, 'decoded_token': 'You'}, '198': {'logprob': -0.6795992255210876, 'rank': 1, 'decoded_token': '\n'}}, {'525': {'logprob': -6.358825206756592, 'rank': 51, 'decoded_token': ' are'}, '220': {'logprob': -3.327575206756592, 'rank': 1, 'decoded_token': ' '}}, {'264': {'logprob': -6.20308780670166, 'rank': 27, 'decoded_token': ' a'}, '198': {'logprob': -1.0780880451202393, 'rank': 1, 'decoded_token': '\n'}}, {'10950': {'logprob': -7.834292411804199, 'rank': 408, 'decoded_token': ' helpful'}, '220': {'logprob': -3.80304217338562, 'rank': 1, 'decoded_token': ' '}}, {'17847': {'logprob': -6.715806484222412, 'rank': 71, 'decoded_token': ' assistant'}, '198': {'logprob': -3.465806484222412, 'rank': 1, 'decoded_token': '\n'}}, {'13': {'logprob': -6.72281551361084, 'rank': 80, 'decoded_token': '.'}, '198': {'logprob': -2.97281551361084, 'rank': 1, 'decoded_token': '\n'}}, {'151645': {'logprob': -17.618621826171875, 'rank': 125489, 'decoded_token': '<|im_end|>'}, '358': {'logprob': -2.8217475414276123, 'rank': 1, 'decoded_token': ' I'}}, {'198': {'logprob': -0.002514536026865244, 'rank': 1, 'decoded_token': '\n'}}, {'151644': {'logprob': -17.267229080200195, 'rank': 129049, 'decoded_token': '<|im_start|>'}, '40': {'logprob': -3.282853841781616, 'rank': 1, 'decoded_token': 'I'}}, {'872': {'logprob': -26.747129440307617, 'rank': 73916, 'decoded_token': 'user'}, '198': {'logprob': -0.004942698869854212, 'rank': 1, 'decoded_token': '\n'}}, {'198': {'logprob': -3.173706531524658, 'rank': 1, 'decoded_token': '\n'}}, {'3838': {'logprob': -3.569950819015503, 'rank': 3, 'decoded_token': 'What'}, '40': {'logprob': -3.319950819015503, 'rank': 1, 'decoded_token': 'I'}}, {'374': {'logprob': -0.6269921660423279, 'rank': 1, 'decoded_token': ' is'}}, {'279': {'logprob': -0.12053166329860687, 'rank': 1, 'decoded_token': ' the'}}, {'6722': {'logprob': -5.329067230224609, 'rank': 39, 'decoded_token': ' capital'}, '6672': {'logprob': -2.1415669918060303, 'rank': 1, 'decoded_token': ' difference'}}, {'315': {'logprob': -0.42458948493003845, 'rank': 1, 'decoded_token': ' of'}}, {'9625': {'logprob': -1.9338321685791016, 'rank': 2, 'decoded_token': ' France'}, '279': {'logprob': -1.8713321685791016, 'rank': 1, 'decoded_token': ' the'}}, {'30': {'logprob': -1.8609070777893066, 'rank': 3, 'decoded_token': '?'}, '5267': {'logprob': -0.9859070777893066, 'rank': 1, 'decoded_token': '?\n'}}, {'151645': {'logprob': -0.012313065119087696, 'rank': 1, 'decoded_token': '<|im_end|>'}}, {'198': {'logprob': -0.015673426911234856, 'rank': 1, 'decoded_token': '\n'}}, {'151644': {'logprob': -0.00029404606902971864, 'rank': 1, 'decoded_token': '<|im_start|>'}}, {'77091': {'logprob': -24.26805877685547, 'rank': 13010, 'decoded_token': 'assistant'}, '198': {'logprob': -0.0024345065467059612, 'rank': 1, 'decoded_token': '\n'}}, {'198': {'logprob': -14.298739433288574, 'rank': 247, 'decoded_token': '\n'}, '785': {'logprob': -0.017489846795797348, 'rank': 1, 'decoded_token': 'The'}}], kv_transfer_params=None)
