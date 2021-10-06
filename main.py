from unet import UNet
import torch

model = UNet(3, 10)
x = torch.randn(8, 3, 30, 30)
y = model(x)

