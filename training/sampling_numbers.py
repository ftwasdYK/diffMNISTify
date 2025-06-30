import torch
from train_ddpm_conditional import Diffusion
from models import UNet_conditional
from utils import save_images
import os

if not os.path.exists("./diffMNISTify/data/gen_data/train"):
        os.makedirs("./diffMNISTify/data/gen_data/train")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet_conditional(num_classes=10).to(DEVICE)
ckpp = torch.load('./checkpoints/DDPM_conditional/ddpm.pth')
model.load_state_dict(ckpp['ema_model_state_dict'])
labels = torch.arange(10).to(DEVICE)
ddpm = Diffusion(noise_steps=400, beta_start=1e-4, beta_end=0.02, img_size=32, device=DEVICE)

for j, _ in enumerate(range(200)):
    cfg = [1.5, 2.5, 3, 3.7, 5, 8]
    for c in cfg:
        x = ddpm.sample(model, 10, labels, c).to(DEVICE)
        for i, xi in enumerate(x):
            path = f'./data/gen_data/train/ema_x{i}_{c}_it{j}.png'
            save_images(xi, path)

model.load_state_dict(ckpp['model_state_dict'])
for j, _ in enumerate(range(200)):
    cfg = [1.5, 2.5, 3, 3.7, 5, 8]
    for c in cfg:
        x = ddpm.sample(model, 10, labels, c).to(DEVICE)
        for i, xi in enumerate(x):
            path = f'./data/gen_data/train/x{i}_{c}_it{j}.png'
            save_images(xi, path)