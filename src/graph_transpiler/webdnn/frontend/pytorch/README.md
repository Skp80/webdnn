# PyTorch frontend

**under development**

## Example

```python
import torch, torchvision
from webdnn.frontend.pytorch import PyTorchConverter


model = torchvision.models.alexnet(pretrained=True)
dummy_input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))

graph = PyTorchConverter().convert(model, dummy_input)
```
