# Math internal case selection

## Validation

- Total math paraphrasing items available: 50
- Selected rows: 44
- Cross-model shared high rows: 7
- Shared selection rule: strict top-20 intersection had fewer than 5 cases, so shared cases use all-branch above-median drift plus best mean rank

## Per-branch coverage

- GPT/main high20 selected: 20
- GPT/main low10 selected: 10
- Qwen high20 selected: 20
- Qwen low10 selected: 10
- Llama high20 selected: 20
- Llama low10 selected: 10

## Completeness checks

- Rows missing original/paraphrased prompt: 0
- Rows missing at least one branch output sample: 0

## Top cross-model shared candidates

- math_reasoning_4506: mean_ncp=0.216196, mean_rank=2.00, high20_branch_count=3
- math_reasoning_6807: mean_ncp=0.060980, mean_rank=11.00, high20_branch_count=3
- math_reasoning_1121: mean_ncp=0.054350, mean_rank=11.67, high20_branch_count=3
- math_reasoning_2873: mean_ncp=0.063004, mean_rank=12.67, high20_branch_count=2
- math_reasoning_6164: mean_ncp=0.057303, mean_rank=13.00, high20_branch_count=2
- math_reasoning_11759: mean_ncp=0.052488, mean_rank=14.67, high20_branch_count=3
- math_reasoning_5408: mean_ncp=0.049317, mean_rank=16.00, high20_branch_count=2
