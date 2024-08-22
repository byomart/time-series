# time-series

- Input: [32, 10, 1]
- After LTSM layer: [32, 10, 64]
- After last sequence selection + dropout: [32, 64]
- After fc1: [32, 32]
- After fc2: [32, 1]


- version 0: 1 epoch
- version 1: 3 epochs
- version 2: 5 epochs  (mantengo 5 epochs a partir de aquí)
- version 3: bidireccional LTSM with:
    - Input: [32, 10, 1]
    - After LTSM layer: [32, 10, 128]
    - After last sequence selection + dropout: [32, 128]
    - After fc1: [128, 64]
    - After fc2: [64, 32]
    - After fc3: [32, 1]
- version 4: 3 capas LTSM (hasta ahora habíamos usado 1)
- version 5: 5 capas LTSM (best option)
- version 6: 10 capas LTSM (+ LOSS = OVERFITTING)
- version 7: 5 LTSM y lr=0.01 (antes lr=0.0001)
- version 8: 5 LTSM y lr=0.001
