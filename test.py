from src.model import PatchCNN
import numpy as np
import torch
device = torch.device('cpu')
model = PatchCNN()

X = np.random.rand(2, 3, 48, 48)
tensor = torch.from_numpy(X).to(device=device, dtype=torch.float)
print(model(tensor))
