import torch
import os
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=5, mode='min'):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_ls, model, optimizer, epoch):
        if self.best is None:
            self.best = current_ls
            return False

        if self.mode == 'min' and current_ls < self.best:
            self.best = current_ls
            self.counter = 0
        elif self.mode == 'max' and current_ls > self.best:
            self.best = current_ls
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

class BaseTrain:
    def __init__(
            self, 
            model, 
            train_loader, val_loader, 
            loss_fn, optimizer, 
            epochs:int, 
            device:str, 
            checkpoint_dir:str='./checkpoints', 
            patience=10, 
            scheduler=None,
            gen_data=False
            ):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history = {'train_loss': [], 'val_loss': []}
        self.epochs = epochs
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.early_stop = EarlyStopping(patience=patience, mode='min')
        self.gen = gen_data

    def train_epoch(self):
        pass

    def val_epoch(self):
        pass

    def train_nn(self):
        for epoch in range(self.epochs):
            self.train_epoch()
            self.val_epoch()
            train_loss = self.history['train_loss'][-1] 
            val_loss = self.history['val_loss'][-1]
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:>7f}, Val Loss: {val_loss:>7f}")
            
            
            # Early stopping
            if self.early_stop(val_loss, self.model, self.optimizer, epoch):
                print("-"*10)
                print("---Early stopping---")
                print("-"*10)
                # Save checkpoint
                checkp_dir = self.save_checkpoint(epoch + 1)
                self.plot_history()
                return self.history, checkp_dir
            self.scheduler.step()
   
        checkp_dir = self.save_checkpoint(epoch + 1)
        self.plot_history()
        return self.history, checkp_dir
    
    def plot_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def save_checkpoint(self, epoch):
        # take models name and save it
        model_name = self.model.__class__.__name__
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{model_name}')
        
        # check if the folder with model's name exist
        i = 1
        while os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(self.checkpoint_dir, f'{model_name}_{i}')
            i += 1
        os.makedirs(checkpoint_path)
        
        gen_dir = 'gen' if self.gen else '' 
        checkpoint_path = os.path.join(checkpoint_path, f'ckp_epoch_{epoch}_{gen_dir}.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.history
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['loss']
        print(f"Checkpoint loaded from {checkpoint_path}")


class MultiClassificationTrainer(BaseTrain):
    """
    Classification Train Class
    """

    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, epochs, device, checkpoint_dir = './checkpoints', patience=10, scheduler=None, verbose=False, gen_data=False):   
        super().__init__(model, train_loader, val_loader, loss_fn, optimizer, epochs, device, checkpoint_dir, patience, scheduler, gen_data=gen_data)
        self.verbose = verbose

    def train_epoch(self):
        size = len(self.train_loader)
        self.model.train()
        train_loss = 0.0
        for batch, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            
            if self.verbose:
                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(inputs)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size*len(inputs):>5d}]")

        self.history['train_loss'].append(train_loss / len(self.train_loader))

    def val_epoch(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, data in enumerate(self.val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
        self.history['val_loss'].append(val_loss / len(self.val_loader))