# Factual QA Paraphrasing Position Summary

This table checks whether `factual_qa + paraphrasing` is the largest noise-corrected drift cell in each available branch.

- Branches checked: 3
- Branches where target ranks first: 3
- Branches where target does not rank first: 0

## Conclusion

PASS. `factual_qa + paraphrasing` ranks first in ChatGPT/main, Qwen, and Llama summaries.

## Summary Table

| branch   | model_label         |   target_rank_descending |   target_mean_noise_corrected_drift | next_largest_cell                 |   next_largest_mean_noise_corrected_drift |   gap_from_next_largest |   ratio_to_non_factual_paraphrasing_mean |   ratio_to_other_factual_qa_perturbation_mean |
|:---------|:--------------------|-------------------------:|------------------------------------:|:----------------------------------|------------------------------------------:|------------------------:|-----------------------------------------:|----------------------------------------------:|
| outputs  | ChatGPT GPT-4o mini |                        1 |                            0.186192 | open_ended_writing + paraphrasing |                                  0.135086 |                0.051106 |                                  3.01801 |                                       11.7772 |
| qwen     | Qwen                |                        1 |                            0.267907 | open_ended_writing + paraphrasing |                                  0.181581 |                0.086326 |                                  2.75669 |                                       12.1292 |
| llama    | Llama               |                        1 |                            0.230321 | open_ended_writing + paraphrasing |                                  0.112716 |                0.117605 |                                  3.55149 |                                       12.4779 |
