import importlib.util
import sys
import time
import types

sp_module = importlib.util.spec_from_file_location(
    "nanovllm.sampling_params", "nanovllm/sampling_params.py"
)
sp = importlib.util.module_from_spec(sp_module)
sp_module.loader.exec_module(sp)
SamplingParams = sp.SamplingParams

# Create a stub nanovllm module to satisfy imports in speculative.py
nanovllm_stub = types.ModuleType('nanovllm')
setattr(nanovllm_stub, 'LLM', object)
setattr(nanovllm_stub, 'SamplingParams', SamplingParams)
sys.modules['nanovllm'] = nanovllm_stub

spec_module = importlib.util.spec_from_file_location(
    "nanovllm.speculative", "nanovllm/speculative.py"
)
spec = importlib.util.module_from_spec(spec_module)
sys.modules['nanovllm.speculative'] = spec
spec_module.loader.exec_module(spec)
speculative_generate = spec.speculative_generate
SpeculativeParams = spec.SpeculativeParams

class ToyTokenizer:
    eos_token_id = 49

    def encode(self, text):
        return [ord(c) % 50 for c in text]

    def decode(self, ids):
        return ''.join(chr(i + 65) for i in ids)

class ToyLLM:
    def __init__(self, overhead_loops=100_000, per_token_loops=10_000):
        self.tokenizer = ToyTokenizer()
        self.overhead_loops = overhead_loops
        self.per_token_loops = per_token_loops

    def generate(self, prompts, sampling_params, use_tqdm=True):
        if not isinstance(prompts, list):
            prompts = [prompts]
        outputs = []
        for _ in prompts:
            s = 0
            for i in range(self.overhead_loops):
                s = (s + i) % 1000
            tokens = []
            for _ in range(sampling_params.max_tokens):
                for j in range(self.per_token_loops):
                    s = (s + j) % 1000
                tokens.append(s % 50)
            outputs.append({"text": self.tokenizer.decode(tokens), "token_ids": tokens})
        return outputs

def naive_generate(llm, prompts, sampling_params):
    outputs = [[] for _ in prompts]
    for _ in range(sampling_params.max_tokens):
        step = llm.generate(prompts, SamplingParams(max_tokens=1), use_tqdm=False)
        for out, s in zip(outputs, step):
            out.extend(s["token_ids"])
    return outputs

def test_speculative_faster():
    target = ToyLLM(overhead_loops=200_000, per_token_loops=50_000)
    draft = ToyLLM(overhead_loops=50_000, per_token_loops=10_000)
    prompts = ["hello"]
    sp = SamplingParams(max_tokens=8)

    start = time.perf_counter()
    naive_generate(target, prompts, sp)
    baseline = time.perf_counter() - start

    start = time.perf_counter()
    speculative_generate(target, draft, prompts, sp, SpeculativeParams(draft_steps=4), use_tqdm=False)
    spec_time = time.perf_counter() - start

    assert spec_time < baseline
