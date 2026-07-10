# LOCO report — preference-labeled sigma predictor

## brightness+tnoise
- interval-MAE: model 0.00463 | best-constant 0.00429 | fixed-0.07 0.02095
- wins/losses/ties vs best-constant: 3/17/1  sign_p=0.9998
- spearman(pred, label) = 0.223   boundary-binding frac: [0.0, 0.14285714285714285]

## brightness-only
- interval-MAE: model 0.00442 | best-constant 0.00429 | fixed-0.07 0.02095
- wins/losses/ties vs best-constant: 3/17/1  sign_p=0.9998
- spearman(pred, label) = 0.223   boundary-binding frac: [0.0]

## per-clip (full model)
|stem|label|lo,hi|pred|const|
|--|--|--|--|--|
|6174|0.01|0.01,0.02|0.0176|0.05|
|8742|0.02|0.01,0.03|0.0300|0.05|
|8656|0.01|0.01,0.01|0.0282|0.05|
|5042|0.05|0.04,0.06|0.0423|0.05|
|0487|0.05|0.05,0.05|0.0603|0.05|
|5281|0.05|0.05,0.05|0.0625|0.05|
|4165|0.05|0.05,0.06|0.0481|0.05|
|7052|0.05|0.05,0.06|0.0492|0.05|
|0265|0.05|0.05,0.05|0.0485|0.05|
|0909|0.05|0.05,0.05|0.0479|0.05|
|1352|0.05|0.05,0.05|0.0522|0.05|
|1389|0.05|0.05,0.05|0.0522|0.05|
|1455|0.05|0.05,0.05|0.0424|0.05|
|3265|0.05|0.05,0.05|0.0582|0.05|
|4378|0.06|0.05,0.06|0.0457|0.05|
|4848|0.06|0.05,0.06|0.0441|0.05|
|4849|0.06|0.05,0.06|0.0428|0.05|
|5280|0.05|0.05,0.05|0.0595|0.05|
|5686|0.05|0.05,0.05|0.0517|0.05|
|5863|0.05|0.05,0.06|0.0488|0.05|
|8267|0.05|0.05,0.05|0.0502|0.05|

## decision inputs: gate_full=False tnoise_earns=False gate_bonly=False
SHIP: NONE (gate failed)
