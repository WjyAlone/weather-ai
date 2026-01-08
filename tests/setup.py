import torch

print(f'Torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Number of CUDA devices: {torch.cuda.device_count()}')
x = torch.tensor([10.0])
x = x.cuda()
y = torch.rand(3, 4)
y = y.cuda()
z = x + y
print(f'Result tensor on CUDA: {z}')
