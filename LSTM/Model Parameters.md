| Dataset        | nx-nd-ny    | WindowLength  | #layer&HiddenSize    | ro-rhidden    |  BatchSize(PRE-INN) | Train-Val-Test| Normalization |Max Epoch(PRE-INN) |
|----------------|-------------|---------------|----------------------|---------------|---------------------|---------------|----------------|------------------|
| Hair Dryer     | 2−2−3       | 30            | 3(35,35)             | 1-0.2           | 32-16               | 40−10−50      | z-score        | 100-20           |
| Heat Exchanger | 0−0−3       | 80            | 3(48,48)             | 1-0.2           |  128-64              | 20−5−75       | min-max        | 50-20           |    
| MR-Damper      | 2−0−1       | 40            | 3(48,48)             | 1-0.2     |  64-64              | 54 − 13 −33   |   z-score      | 100-20           |  

| Dataset        | lr(PRE-INN) | lr scheduler(PRE-INN)                | patience(PRE-INN)   | min_delta(PRE-INN) |
|----------------|-------------|--------------------------------------|---------------------|--------------------|
| Hair Dryer     | 5e-3 - 1e-2 | 20 Epoch & lr*0.5- 10 Epoch & lr*0.5 |      20-10           | 0.001-0.005       |
| Heat Exchanger | 5e-3 - 1e-2 | 20 Epoch & lr*0.5- 10 Epoch & lr*0.5 |      20-10           |      0.001-0.005  |                   
| MR-Damper      | 1e-3 - 5e-3 | No scheduler used- 10 Epoch & lr*0.5 | 20-5               |      0.0005-0.0005  |        

