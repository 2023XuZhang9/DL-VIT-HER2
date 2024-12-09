import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model_path, patience=8, verbose=False):
        """
        Args:
            model_path (str): Path to save the model.
            patience (int): How long to wait after the last time validation loss improved. Default: 8
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.model_path = model_path

    def __call__(self, val_loss, model):
        """
        Check if validation loss has improved and save the model if it has.
        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model to be saved.
        """
        score = -val_loss  # Use negative loss because lower validation loss is better
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.model_path}')
        torch.save(model.state_dict(), self.model_path)
        self.val_loss_min = val_loss
