# Controlled synthetic experiment

Generated English business documents test execution, leakage controls and learning. They do not establish effectiveness on real organizational work. Loss includes declared scripted request and hold costs.

| Policy / constructor | n | Total loss | Requests | Hold | False handoff |
|---|---:|---:|---:|---:|---:|
| checklist/rules | 120 | 75.918 | 0.333 | 0.0% | 0.0% |
| ask_all/rules | 120 | 38.457 | 1.042 | 0.0% | 0.0% |
| uncertainty/rules | 120 | 38.457 | 1.042 | 0.0% | 0.0% |
| one_step/rules | 120 | 35.332 | 0.917 | 0.0% | 0.0% |
| reference/rules | 120 | 34.082 | 0.875 | 0.0% | 0.0% |
| learned/rules | 120 | 64.299 | 0.458 | 0.0% | 0.0% |
| checklist/learned | 120 | 87.702 | 0.300 | 0.0% | 0.0% |
| ask_all/learned | 120 | 107.494 | 1.175 | 0.0% | 0.0% |
| uncertainty/learned | 120 | 107.494 | 1.175 | 0.0% | 0.0% |
| one_step/learned | 120 | 87.729 | 0.300 | 0.0% | 0.0% |
| reference/learned | 120 | 87.729 | 0.300 | 0.0% | 0.0% |
| learned/learned | 120 | 89.994 | 0.425 | 0.0% | 0.0% |
| oracle/full-information-bound | 120 | 0.000 | 0.000 | 0.0% | 0.0% |
| learned/learned/no-evidence | 120 | 67.906 | 0.467 | 0.0% | 100.0% |
| learned/learned/no-type | 120 | 110.240 | 0.467 | 0.0% | 0.0% |
| learned/learned/no-impact | 120 | 89.994 | 0.425 | 0.0% | 0.0% |
| learned/learned/no-update | 120 | 185.827 | 0.425 | 16.7% | 0.0% |
| learned/learned/noise-0.1 | 120 | 89.994 | 0.425 | 0.0% | 4.2% |
| learned/learned/noise-0.2 | 120 | 89.994 | 0.425 | 0.0% | 5.8% |

## Paired differences

learned/learned vs reference/learned: 2.264, 95% source-bundle bootstrap CI [-6.351, 11.507], 24 bundles.

learned/learned vs one_step/learned: 2.264, 95% source-bundle bootstrap CI [-6.351, 11.507], 24 bundles.

learned/rules vs reference/rules: 30.217, 95% source-bundle bootstrap CI [13.546, 48.690], 24 bundles.

learned/rules vs one_step/rules: 28.967, 95% source-bundle bootstrap CI [11.606, 47.802], 24 bundles.

Ablations and response-error stress tests here are inference diagnostics. Retrained ablation studies require separate runs.
