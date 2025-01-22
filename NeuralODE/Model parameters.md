| Dataset        | nx-nd-ny    | WindowLength  | #layer&HiddenSize    | ro-rhidden    |  BatchSize(PRE-INN) | Train-Val-Test| Normalization |Max Epoch(PRE-INN) |
|----------------|-------------|---------------|----------------------|---------------|---------------------|---------------|----------------|------------------|
| Hair Dryer     | 2−2−3       | 30            | 3(40,40)             | 1-1           | 16-16               | 40−10−50      | z-score        | 100-30           |
| Heat Exchanger | 0−0−3       | 80            | 3(35,10)             | 1-1           |  64-64              | 20−5−75       | min-max        | 200-30           |    
| MR-Damper      | 2−0−1       | 40            | 2(32)                | 0.75-0.75     |  64-64              | 54 − 13 −33   |   z-score      | 100-30           |  

| Dataset        | lr(PRE-INN) | lr scheduler(PRE-INN)                | patience(PRE-INN)   | min_delta(PRE-INN) |
|----------------|-------------|--------------------------------------|---------------------|--------------------|
| Hair Dryer     | 1e-2 - 1e-2 | 30 Epoch & lr*0.5- 20 Epoch & lr*0.5 |      20-5           | 0.0005-0.005       |
| Heat Exchanger | 5e-3 - 1e-2 | 75 Epoch & lr*0.5- 20 Epoch & lr*0.5 |      20-5           |      0.0005-0.005  |                   
| MR-Damper      | 1e-3 - 1e-2 | No scheduler used- 20 Epoch & lr*0.1 | 100-5               |      0.0005-0.005  |        

