import torch
from torch.distributions import Categorical

probs = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                      [0.4, 0.3, 0.2, 0.1]])
dist = Categorical(probs)

samples = dist.sample((3,))
print(samples)