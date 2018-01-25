# CyclicLR_Scheduler_PyTorch
A PyTorch implementation of Cyclical Learning Rates

Please refer to [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) for more details

# Usage
```python
from cyclic_lr_scheduler import CyclicLR

optimizer = Whatever optimizer you want

scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size=10, mode=decay_strategy)
```

+ three options for decay_strategy: 'triangular', 'triangular2', 'exp_range'
+ step_size denotes the number of epoch rather than iteration