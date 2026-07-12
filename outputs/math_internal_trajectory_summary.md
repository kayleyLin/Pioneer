# Math trajectory divergence probe

Source: `outputs/math_internal_logprob_probe_by_item.csv` deterministic Together generations.

## Validation

- Rows compared: 44
- Unique items: 44
- Rows with empty original final answer: 0
- Rows with empty paraphrase final answer: 0

## Divergence summary

- all: n=44, operation mismatch=17, first numeric expression mismatch=20, early-number mismatch=41, final-answer mismatch=30, mean output F1=0.540
- shared_high: n=7, operation mismatch=3, first numeric expression mismatch=4, early-number mismatch=7, final-answer mismatch=6, mean output F1=0.423
- non_shared: n=37, operation mismatch=14, first numeric expression mismatch=16, early-number mismatch=34, final-answer mismatch=24, mean output F1=0.562

## Strongest trajectory divergences

- math_reasoning_2873: F1=0.267, op=multiplication->algebra_equation, final=72->10, groups=gpt_high20;qwen_high20;cross_model_shared_high
- math_reasoning_338: F1=0.277, op=geometry->geometry, final=117->30, groups=gpt_low10;qwen_high20;llama_high20
- math_reasoning_11297: F1=0.286, op=subtraction->subtraction, final=-1->-6, groups=gpt_high20;qwen_high20
- math_reasoning_10586: F1=0.289, op=unknown->probability, final=5->10, groups=qwen_high20;llama_high20
- math_reasoning_4506: F1=0.299, op=geometry->geometry, final=13->4, groups=gpt_high20;qwen_high20;llama_high20;cross_model_shared_high
- math_reasoning_5201: F1=0.303, op=subtraction->algebra_equation, final=0->7, groups=gpt_low10;llama_high20
- math_reasoning_3251: F1=0.305, op=subtraction->subtraction, final=0->-3, groups=qwen_low10;llama_low10
- math_reasoning_6807: F1=0.324, op=addition->algebra_equation, final=3->3, groups=gpt_high20;qwen_high20;llama_high20;cross_model_shared_high
- math_reasoning_2817: F1=0.327, op=algebra_equation->subtraction, final=2->75, groups=gpt_high20;llama_low10
- math_reasoning_5912: F1=0.378, op=geometry->division, final=2->3/2, groups=gpt_high20;qwen_high20
