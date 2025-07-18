import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2
from typing import Tuple

from utils import ChannelRepeat, prepare_data, save_confusion_matrix_plot, get_results
from models import CustomResNet50, CustomViT
from training_loops import MultiClassificationTrainer
import pandas as pd
import os

if not os.path.exists("./training/results"):
        os.makedirs("./training/results")

if not os.path.exists("./training/figures"):
        os.makedirs("./training/figures")

BATCH_SIZE = 16
EPOCHS = 2
LR = 1e-2
WEIGHT_DECAY = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NN_MODEL = CustomResNet50(pretrained=False).to(DEVICE) # CustomViT().to(DEVICE)# 
GEN_DATA = True
# Optimizer
optimizer = torch.optim.AdamW(NN_MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5, gamma=0.5) 
print('Total trainable parameters', NN_MODEL.count_total_parameters())


transform_train = v2.Compose([
        v2.ToTensor(),
        ChannelRepeat(), 
        NN_MODEL.get_transformation(),
        v2.RandomHorizontalFlip(p=0.2),
        v2.RandomRotation(degrees=30),
        v2.RandomVerticalFlip(p=0.2),
        v2.GaussianNoise()
    ])

transform_inference = v2.Compose([
        v2.ToTensor(),
        ChannelRepeat(), 
        NN_MODEL.get_transformation(),
    ])

train_loader, val_loader, test_loader = prepare_data(
    transform_tr=transform_train,
    transform_infer=transform_inference,
    batch_size=BATCH_SIZE,
    gen_data=GEN_DATA
)

nn_m = MultiClassificationTrainer(
    NN_MODEL,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    EPOCHS,
    DEVICE,
    scheduler=scheduler,
    verbose=True,
    gen_data=GEN_DATA
)

hist, checkp_dir = nn_m.train_nn()

# load the best model
nn_m.load_checkpoint(checkp_dir)

##### Evaluation Phase #####   
results, conf = get_results(NN_MODEL, test_loader)

# save results
conf_matrix_df = pd.DataFrame(conf.cpu().numpy())
results_df = pd.DataFrame(results)
print(conf_matrix_df)
print(results_df)
prefix = f'{NN_MODEL.__class__.__name__}_lr{LR}_wd{WEIGHT_DECAY}_bs{BATCH_SIZE}_epochs{EPOCHS}'
gen_dir = 'gen_data' if GEN_DATA else ''
results_df.to_csv(f"./training/results/test_results_{prefix}_{gen_dir}.csv", index=False)
conf_matrix_df.to_csv(f"./training/results/confusion_matrix_{prefix}_{gen_dir}.csv", index=False)

save_confusion_matrix_plot(conf, f'./training/figures/confusion_matrix_{prefix}_{gen_dir}.png')