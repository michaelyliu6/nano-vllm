import os
from nanovllm import LLM, SamplingParams, speculative_generate, SpeculativeParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    draft_path = os.path.expanduser("~/huggingface/TinyLLM/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    draft_llm = LLM(draft_path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    # Standard decoding
    outputs = llm.generate(prompts, sampling_params)

    # Speculative decoding using a smaller draft model
    spec_outputs = speculative_generate(
        llm,
        draft_llm,
        prompts,
        sampling_params,
        SpeculativeParams(draft_steps=4),
    )

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    # Show speculative decoding results
    for prompt, output in zip(prompts, spec_outputs):
        print("\n")
        print(f"[Speculative] Prompt: {prompt!r}")
        print(f"[Speculative] Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
