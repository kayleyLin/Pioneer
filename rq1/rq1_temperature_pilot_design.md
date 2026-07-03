# RQ1 Temperature Pilot Design

## 1. Purpose

This pilot tests whether the temperature setting substantially changes the sampling-noise baseline in RQ1.

The current RQ1 design uses repeated generations from the same prompt to estimate ordinary output variation. Because temperature controls the randomness of model decoding, it may directly affect this baseline. Therefore, before selecting a final temperature setting for the formal RQ1 experiment, a small calibration trial should compare baseline drift across multiple temperature values.

The main question for this pilot is:

```text
How much does the sampling-noise baseline change when temperature changes, while all other generation settings are held constant?
```

This is a parameter-calibration experiment, not the main RQ1 perturbation experiment.

## 2. Experimental Logic

The experiment isolates temperature as the independent variable.

All other settings should remain fixed:

```text
model = gpt-4o-mini
top_p = 0.9
max_output_tokens = not set
system_prompt = "You are a helpful assistant. Answer the user's prompt directly."
prompt set = same calibration prompts across all temperature conditions
similarity metric = Sentence-BERT cosine similarity
```

The logic is:

```text
If the same prompts are used under different temperature values,
and all other generation settings are fixed,
then differences in within-prompt output drift can be attributed mainly to temperature.
```

## 3. Prompt Set

Use the existing RQ1 calibration prompt set:

```text
prompts/rq1_calibration_prompts.csv
```

Current calibration set:

```text
2 prompts per task type
4 task types
8 prompts total
```

This set is appropriate for a temperature pilot because it is small enough to run cheaply but still covers all task types.

The full 20-prompt RQ1 sample should not be used for the first temperature trial unless the pilot shows that temperature has a major effect and the experiment needs confirmation on a larger sample.

## 4. Temperature Conditions

The temperature parameter is not limited to one decimal place. It is a numeric decoding parameter, so values such as `0.75` or `0.85` can also be used. However, the pilot does not need to test every possible value. The purpose is to choose representative points that can reveal the general trend and then inspect the region near the current setting.

The first proposed temperature set was a coarse sweep:

```text
0.0, 0.3, 0.7, 1.0
```

However, this spacing is too wide if the goal is to decide whether the current pilot setting, `temperature = 0.7`, should be kept or slightly adjusted. In that case, the experiment should include more values near the current setting.

Recommended revised temperature values:

| Condition | Temperature | Reason |
|---|---:|---|
| Near-deterministic baseline | 0.0 | Tests the minimum-randomness condition |
| Conservative sampling | 0.3 | Tests mild output variation |
| Moderate sampling | 0.5 | Bridges the gap between low and current settings |
| Current pilot setting | 0.7 | Matches the current RQ1 setup |
| Slightly higher than current | 0.8 | Tests whether drift increases immediately above the current setting |
| High but still common | 0.9 | Tests stronger variation without jumping directly to 1.0 |
| Upper comparison point | 1.0 | Tests high-randomness generation |

This revised design is better because it combines a broad range with a local sweep around the current setting. The values `0.8` and `0.9` are especially useful because they show whether the sampling-noise baseline changes gradually or sharply as temperature increases above `0.7`.

If API cost becomes a concern, the reduced version should be:

```text
0.3, 0.5, 0.7, 0.8, 0.9
```

This reduced version removes the two endpoints while still focusing on the practical decision range.

The current project already has calibration generations at `temperature = 0.7`. Those outputs can be reused as the 0.7 condition if the same prompt set and generation settings were used.

## 5. Repeated Generations Per Prompt

Recommended setting:

```text
n = 5 generations per prompt per temperature
```

Reason:

```text
With n = 5, each prompt produces 10 pairwise similarity comparisons.
This gives a more stable estimate than n = 3 while keeping API cost manageable.
```

Expected output count for the full revised design:

```text
8 prompts * 7 temperatures * 5 generations = 280 outputs
```

If using the existing 0.7 calibration outputs, only the other six temperature conditions need to be newly generated:

```text
8 prompts * 6 new temperatures * 5 generations = 240 new outputs
```

Expected output count for the reduced design:

```text
8 prompts * 5 temperatures * 5 generations = 200 outputs
```

If using the existing 0.7 calibration outputs for the reduced design:

```text
8 prompts * 4 new temperatures * 5 generations = 160 new outputs
```

## 6. Measurement

For each prompt and temperature condition:

```text
1. Generate 5 outputs from the same prompt.
2. Encode each output using Sentence-BERT.
3. Compute pairwise cosine similarity between all output pairs.
4. Average the pairwise similarities.
5. Convert similarity into drift:

   sampling_noise_drift = 1 - mean_within_prompt_similarity
```

For each task type and temperature:

```text
task_temperature_drift = average sampling_noise_drift across the 2 prompts in that task type
```

For the overall temperature effect:

```text
overall_temperature_drift = average sampling_noise_drift across all 8 prompts
```

## 7. Analysis Plan

The pilot should produce three main tables.

### 7.1 Overall Temperature Table

| Temperature | Mean similarity | Mean drift | Interpretation |
|---:|---:|---:|---|
| 0.0 | TBD | TBD | Low-randomness baseline |
| 0.3 | TBD | TBD | Mild randomness |
| 0.5 | TBD | TBD | Moderate randomness |
| 0.7 | TBD | TBD | Current setting |
| 0.8 | TBD | TBD | Slightly above current setting |
| 0.9 | TBD | TBD | High but common setting |
| 1.0 | TBD | TBD | Higher randomness |

This table answers whether higher temperature generally increases output drift.

### 7.2 Temperature By Task Type Table

| Task type | temp=0.0 | temp=0.3 | temp=0.5 | temp=0.7 | temp=0.8 | temp=0.9 | temp=1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| factual_qa | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| math_reasoning | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| code_generation | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| open_ended_writing | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

This table answers whether the temperature effect is task-dependent.

### 7.3 Decision Table

| Possible result | Interpretation | Decision |
|---|---|---|
| Drift increases smoothly as temperature increases | Temperature strongly controls sampling noise | Choose a moderate setting and report it as calibrated |
| Drift is similar from 0.5 to 0.9 | Current setting is likely acceptable | Keep 0.7 |
| Drift increases sharply after 0.7 | Current setting may be near the upper stable range | Keep 0.7 or lower to 0.5 |
| 1.0 produces much higher drift | High temperature may inflate baseline noise | Avoid using 1.0 for formal RQ1 |
| Task types respond differently | Temperature interacts with task type | Report this as a pilot finding and keep one fixed temperature for comparability |

## 8. Recommended Decision Rule

Use the following practical rule after results are generated:

```text
If temperature = 0.7 produces clearly higher drift than 0.5 without adding methodological value,
then use temperature = 0.5 for the formal experiment.

If temperature = 0.7 is close to 0.5, 0.8, and 0.9 and does not inflate drift substantially,
then keep temperature = 0.7 because it is already used in the current pilot.

Avoid temperature = 1.0 for the formal experiment if it substantially increases baseline drift,
because it may make perturbation effects harder to interpret.
```

The final paper should describe the selected temperature as a calibrated generation setting rather than an arbitrary choice.

## 9. How To Explain This To The Professor

Short explanation:

```text
Because RQ1 relies on repeated generations to estimate a sampling-noise baseline, I need to check whether that baseline is sensitive to the model's temperature setting. I designed a small calibration trial using the same eight prompts across several temperature values. Instead of only using a coarse sweep, I include additional values around the current setting, such as 0.8 and 0.9, to see whether drift changes gradually or sharply near 0.7. For each prompt and temperature, I generate five outputs and compute the within-prompt Sentence-BERT similarity. This allows me to see whether the current temperature setting inflates the noise baseline and whether the effect differs by task type.
```

## 10. Status

Current status:

```text
Experiment designed.
Generation code still needs to be adapted for multiple temperature conditions.
Analysis code still needs to be adapted to summarize drift by temperature.
```
