"""Speculative decoding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nanovllm import LLM, SamplingParams


@dataclass
class SpeculativeParams:
    """Parameters for speculative decoding."""

    draft_steps: int = 4


def speculative_generate(
    target: LLM,
    draft: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    spec_params: SpeculativeParams | None = None,
    use_tqdm: bool = True,
) -> List[dict]:
    """Generate text with speculative decoding.

    This implementation follows the draft/verify algorithm in a simplified
    manner. It repeatedly generates ``draft_steps`` tokens using the ``draft``
    model and verifies them with the ``target`` model. The decoding stops once
    ``sampling_params.max_tokens`` tokens are produced for each prompt.
    """
    if spec_params is None:
        spec_params = SpeculativeParams()

    outputs = []
    for prompt in prompts:
        prompt_ids = target.tokenizer.encode(prompt)
        generated: List[int] = []
        while len(generated) < sampling_params.max_tokens:
            # 1. Draft phase: propose ``draft_steps`` tokens.
            draft_out = draft.generate(
                [prompt_ids + generated],
                SamplingParams(
                    temperature=sampling_params.temperature,
                    max_tokens=min(
                        spec_params.draft_steps,
                        sampling_params.max_tokens - len(generated),
                    ),
                    ignore_eos=sampling_params.ignore_eos,
                ),
                use_tqdm=False,
            )[0]["token_ids"]

            # 2. Verify the proposed tokens with the target model in batch.
            target_out = target.generate(
                [prompt_ids + generated],
                SamplingParams(
                    temperature=sampling_params.temperature,
                    max_tokens=len(draft_out),
                    ignore_eos=sampling_params.ignore_eos,
                ),
                use_tqdm=False,
            )[0]["token_ids"]

            for d_tok, t_tok in zip(draft_out, target_out):
                if len(generated) >= sampling_params.max_tokens:
                    break
                if d_tok == t_tok:
                    generated.append(d_tok)
                else:
                    generated.append(t_tok)
                    break
                if (
                    not sampling_params.ignore_eos
                    and t_tok == target.tokenizer.eos_token_id
                ):
                    break
            if (
                not sampling_params.ignore_eos
                and generated
                and generated[-1] == target.tokenizer.eos_token_id
            ):
                break
        outputs.append(
            {
                "text": target.tokenizer.decode(generated),
                "token_ids": generated,
            }
        )
    return outputs
