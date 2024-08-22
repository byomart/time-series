# time-series

- Input: [32, 10, 1]
- After LTSM layer: [32, 10, 64]
- After last sequence selection + dropout: [32, 64]
- After fc1: [32, 32]
- After fc2: [32, 1]


- version 0: 1 epoch
- version 1: 3 epochs
- version 2: 5 epochs