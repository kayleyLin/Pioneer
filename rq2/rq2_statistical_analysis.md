# RQ2 Statistical Analysis
## Input

Input table: `/Users/wenfenglin/Desktop/Pioneer/rq2/outputs/rq2_formal_available_drift_performance_by_item.csv`

Rows: 150 item-perturbation observations.

Outcome: `absolute_performance_change`.

Main predictor: `noise_corrected_drift`.

Task and perturbation type are treated as categorical factors.


## OLS 1: Drift Only

```text
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:     absolute_performance_change   R-squared:                       0.107
Model:                                     OLS   Adj. R-squared:                  0.101
Method:                          Least Squares   F-statistic:                     7.359
Date:                         Wed, 01 Jul 2026   Prob (F-statistic):            0.00746
Time:                                 17:18:51   Log-Likelihood:                 3.5522
No. Observations:                          150   AIC:                            -3.104
Df Residuals:                              148   BIC:                             2.917
Df Model:                                    1                                         
Covariance Type:                           HC3                                         
=========================================================================================
                            coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------------
Intercept                 0.0563      0.017      3.411      0.001       0.024       0.089
noise_corrected_drift     1.6937      0.624      2.713      0.007       0.470       2.917
==============================================================================
Omnibus:                       79.808   Durbin-Watson:                   1.860
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              303.782
Skew:                           2.068   Prob(JB):                     1.08e-66
Kurtosis:                       8.612   Cond. No.                         20.7
==============================================================================

Notes:
[1] Standard Errors are heteroscedasticity robust (HC3)
```


## OLS 2: Drift By Task Interaction

```text
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:     absolute_performance_change   R-squared:                       0.220
Model:                                     OLS   Adj. R-squared:                  0.193
Method:                          Least Squares   F-statistic:                     3.953
Date:                         Wed, 01 Jul 2026   Prob (F-statistic):            0.00217
Time:                                 17:18:51   Log-Likelihood:                 13.652
No. Observations:                          150   AIC:                            -15.30
Df Residuals:                              144   BIC:                             2.760
Df Model:                                    5                                         
Covariance Type:                           HC3                                         
========================================================================================================================
                                                           coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------------------
Intercept                                                0.0369      0.029      1.282      0.200      -0.020       0.093
C(task_type)[T.factual_qa]                              -0.0026      0.038     -0.070      0.944      -0.077       0.071
C(task_type)[T.math_reasoning]                           0.0221      0.043      0.517      0.605      -0.062       0.106
noise_corrected_drift                                    5.0216      2.300      2.183      0.029       0.513       9.530
noise_corrected_drift:C(task_type)[T.factual_qa]        -3.2252      2.382     -1.354      0.176      -7.893       1.443
noise_corrected_drift:C(task_type)[T.math_reasoning]    -8.4603      2.736     -3.092      0.002     -13.822      -3.098
==============================================================================
Omnibus:                       85.249   Durbin-Watson:                   1.870
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              329.471
Skew:                           2.241   Prob(JB):                     2.86e-72
Kurtosis:                       8.711   Cond. No.                         139.
==============================================================================

Notes:
[1] Standard Errors are heteroscedasticity robust (HC3)
```


## OLS 3: Drift By Task Interaction + Perturbation Controls

```text
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:     absolute_performance_change   R-squared:                       0.264
Model:                                     OLS   Adj. R-squared:                  0.217
Method:                          Least Squares   F-statistic:                     2.919
Date:                         Wed, 01 Jul 2026   Prob (F-statistic):            0.00336
Time:                                 17:18:51   Log-Likelihood:                 18.025
No. Observations:                          150   AIC:                            -16.05
Df Residuals:                              140   BIC:                             14.06
Df Model:                                    9                                         
Covariance Type:                           HC3                                         
========================================================================================================================
                                                           coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------------------
Intercept                                                0.0137      0.039      0.351      0.726      -0.063       0.090
C(task_type)[T.factual_qa]                               0.0027      0.039      0.069      0.945      -0.073       0.079
C(task_type)[T.math_reasoning]                           0.0118      0.044      0.270      0.787      -0.074       0.097
C(perturbation_type)[T.formatting_changes]               0.0129      0.052      0.248      0.804      -0.089       0.115
C(perturbation_type)[T.paraphrasing]                     0.1491      0.075      1.983      0.047       0.002       0.296
C(perturbation_type)[T.reordering]                       0.0114      0.049      0.230      0.818      -0.086       0.108
C(perturbation_type)[T.surface_noise]                   -0.0063      0.038     -0.164      0.870      -0.081       0.069
noise_corrected_drift                                    4.0205      2.425      1.658      0.097      -0.732       8.773
noise_corrected_drift:C(task_type)[T.factual_qa]        -2.6834      2.477     -1.083      0.279      -7.539       2.172
noise_corrected_drift:C(task_type)[T.math_reasoning]    -7.8521      2.755     -2.850      0.004     -13.253      -2.451
==============================================================================
Omnibus:                       78.353   Durbin-Watson:                   1.836
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              268.927
Skew:                           2.086   Prob(JB):                     4.01e-59
Kurtosis:                       8.061   Cond. No.                         150.
==============================================================================

Notes:
[1] Standard Errors are heteroscedasticity robust (HC3)
```


## Mixed-Effects Robustness Check

```text
                              Mixed Linear Model Regression Results
=================================================================================================
Model:                    MixedLM         Dependent Variable:         absolute_performance_change
No. Observations:         150             Method:                     ML                         
No. Groups:               30              Scale:                      0.0470                     
Min. group size:          5               Log-Likelihood:             16.4436                    
Max. group size:          5               Converged:                  Yes                        
Mean group size:          5.0                                                                    
-------------------------------------------------------------------------------------------------
                                                      Coef.  Std.Err.   z    P>|z|  [0.025 0.975]
-------------------------------------------------------------------------------------------------
Intercept                                              0.000    0.048  0.000 1.000  -0.094  0.094
C(task_type)[T.factual_qa]                             0.000    0.047  0.000 1.000  -0.093  0.093
C(task_type)[T.math_reasoning]                         0.000    0.045  0.000 1.000  -0.088  0.088
C(perturbation_type)[T.formatting_changes]             0.012    0.056  0.213 0.831  -0.098  0.122
C(perturbation_type)[T.paraphrasing]                   0.139    0.061  2.294 0.022   0.020  0.258
C(perturbation_type)[T.reordering]                     0.010    0.057  0.172 0.863  -0.101  0.120
C(perturbation_type)[T.surface_noise]                 -0.008    0.056 -0.146 0.884  -0.118  0.102
noise_corrected_drift                                  5.348    1.225  4.365 0.000   2.947  7.749
noise_corrected_drift:C(task_type)[T.factual_qa]      -4.082    1.268 -3.219 0.001  -6.567 -1.596
noise_corrected_drift:C(task_type)[T.math_reasoning] -10.128    1.886 -5.371 0.000 -13.824 -6.432
Group Var                                              0.000    0.009                            
=================================================================================================

```

## Output CSV Files

```text
rq2/outputs/rq2_stats_descriptive_by_task.csv
rq2/outputs/rq2_stats_descriptive_by_perturbation.csv
rq2/outputs/rq2_stats_descriptive_by_task_perturbation.csv
rq2/outputs/rq2_stats_correlations.csv
```
