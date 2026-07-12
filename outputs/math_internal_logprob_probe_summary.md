# Math internal logprob probe

- Model: `meta-llama/Meta-Llama-3-8B-Instruct-Lite`
- Items probed: 44

## Summary

- all: n=44, mean reference logprob delta=nan, negative reference delta count=0, mean generated logprob delta=0.000771
- shared_high: n=7, mean reference logprob delta=nan, negative reference delta count=0, mean generated logprob delta=-0.017636
- non_shared: n=37, mean reference logprob delta=nan, negative reference delta count=0, mean generated logprob delta=0.004254

Definition:

`delta = paraphrased_prompt_condition - original_prompt_condition`.
A negative reference delta means the original Llama continuation is less likely under the paraphrased prompt.
If reference deltas are NaN, the hosted endpoint did not expose prompt echo logprobs for this model; generated-token logprob deltas are then the usable API-only signal.

## Strongest negative reference-likelihood shifts

- Not available: Together did not return prompt echo logprobs for fixed-reference scoring.

## Strongest negative generated-token confidence shifts

- math_reasoning_4807: generated_delta=-0.150102, groups=gpt_high20;llama_high20
- math_reasoning_10586: generated_delta=-0.146466, groups=qwen_high20;llama_high20
- math_reasoning_5408: generated_delta=-0.101031, groups=qwen_high20;llama_high20;cross_model_shared_high
- math_reasoning_3402: generated_delta=-0.081674, groups=gpt_high20;qwen_low10
- math_reasoning_11297: generated_delta=-0.075451, groups=gpt_high20;qwen_high20
- math_reasoning_2891: generated_delta=-0.070040, groups=gpt_low10;qwen_high20
- math_reasoning_7390: generated_delta=-0.067699, groups=qwen_low10
- math_reasoning_4506: generated_delta=-0.063049, groups=gpt_high20;qwen_high20;llama_high20;cross_model_shared_high
- math_reasoning_9403: generated_delta=-0.046053, groups=llama_high20
- math_reasoning_4531: generated_delta=-0.035447, groups=gpt_high20;llama_low10
