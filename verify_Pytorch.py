import torch, torchvision, torchaudio
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("CUDA available?", torch.cuda.is_available())
print("CPU OK:", torch.ones(2,2).sum().item())
