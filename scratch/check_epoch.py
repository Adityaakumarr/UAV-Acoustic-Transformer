import torch
try:
    ckpt = torch.load('checkpoints/last_model.pth', map_location='cpu', weights_only=False)
    print('Last completed epoch:', ckpt.get('epoch', 'Unknown'))
except Exception as e:
    print('Error reading checkpoint:', e)
