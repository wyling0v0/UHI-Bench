# UHI-Bench Temporal and Leakage-Control Protocol

## Default split

The default benchmark protocol is:

> **2015–2022 train / 2023–2025 evaluation**

All supervised model fitting and all fitted preprocessing are restricted to the
training period. This includes normalization means and standard deviations,
missing-value fill statistics, climatologies, learned thresholds, and model
selection quantities. Once fitted, these objects are frozen for evaluation.

## Temporal boundary and context

There is no additional unused buffer or embargo between 2022 and 2023. An
evaluation prediction at time `t` may use values observed strictly before `t`.
Consequently, the earliest 2023 predictions may use late-2022 observations to
fill a 24-hour, 96-hour, or 168-hour context window. Those observations are
inputs only; no evaluation-period target is included in model fitting.

This is an operational forecasting convention, not a claim of complete
temporal independence between adjacent inputs. A stricter embargo can be added
as a sensitivity analysis by dropping the first `context_length` evaluation
hours.

## Imputation masks

- LST cloud-gap evaluation uses cloud-missingness patterns from the held-out
  evaluation period and transfers them to clear evaluation scenes. Mask
  selection does not use hidden pixel magnitudes. Learned imputers are fitted
  on 2015–2022; evaluation masks and hidden values are not used for fitting.
- AirT sparse-reconstruction evaluation uses seeded random spatial masks in the
  held-out evaluation period. Mask locations are sampled independently of the
  hidden target values. When a learned imputer needs synthetic training masks,
  those masks are generated separately on 2015–2022 training windows.
- Scene-wise spatial baselines such as IDW and kriging are transductive by
  design: at each evaluation timestamp they use only the visible pixels from
  that scene to reconstruct its hidden pixels. They do not fit a temporal model
  on evaluation targets.

## Extreme-event detection

The event threshold is fitted from the 2015–2022 target series and frozen for
2023–2025. Above-threshold runs are constructed separately on the two sides of
the split so an event cannot cross the boundary. Sequence classifiers consume
history ending at `t-1` and predict the label at `t`.

## Released target construction

The split above governs downstream benchmark experiments. It must not be used
to infer how an already released target artifact was constructed. In
particular, the current international model-derived AirT-UHI artifact was
created by an earlier residual-model pipeline whose fit period included
2015–2023 German data and whose validation period was 2024. Therefore, results
using its 2023–2025 values are evaluation of downstream generalization, but not
a strictly train-only audit of target construction. This limitation is recorded
in the dataset card and provenance metadata and should be disclosed in the
paper.

The released 1 km LST product is also a derived target. Its RF-TsHARP mapping
was fitted separately for each target year and season/month using the
contemporaneous coarse-resolution LST field, including during 2023–2025. It is
therefore best described as observation-conditioned spatial downscaling, not as
a single 2015–2022-trained temporal model. Downstream benchmark models remain
held out by year, but the target-construction distinction must be reported.
