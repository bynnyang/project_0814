# burn.py
import torch, time

torch.cuda.set_device(0)
x = torch.randn(3000, 3000, device="cuda")

while True:
    x = x @ x
    torch.cuda.synchronize()
    time.sleep(0.01)