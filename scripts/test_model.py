import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from models.cnn1d_model import CNN1D_Wide, VibrationDataset

# Load model
model = CNN1D_Wide()
ckpt = torch.load('./cnn1d_model_new.ckpt', map_location='cpu')
state = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load data
dataset = VibrationDataset('./data')
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Evaluate
correct = 0
total = 0
class_correct = [0, 0]
class_total = [0, 0]

with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for c in [0, 1]:
            mask = (y == c)
            class_correct[c] += (pred[mask] == y[mask]).sum().item()
            class_total[c] += mask.sum().item()

print(f"\nOverall accuracy: {correct/total:.4f} ({correct}/{total})")
print(f"OK  (class 0): {class_correct[0]/class_total[0]:.4f} ({class_correct[0]}/{class_total[0]})")
print(f"NOK (class 1): {class_correct[1]/class_total[1]:.4f} ({class_correct[1]}/{class_total[1]})")