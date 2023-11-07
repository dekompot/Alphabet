import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

LOC = 'data'
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
class_mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
np.random.seed(2024)


def load_data(prefix):
    return (np.load(f'{prefix}/X_train.npy'), np.load(f'{prefix}/y_train.npy'), np.load(f'{prefix}/X_val.npy'),
            np.load(f'{prefix}/y_val.npy'), np.load(f'{prefix}/X_test.npy'))


def transform_imgs(X, mean, std):
    X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    X = transforms.Normalize(mean, std)(X)
    return X


def transform_labels(y):
    y = torch.tensor(y, device=DEVICE)
    y = y.argmax(dim=1)
    return y


def get_loaders_and_weights(batch_size):
    X_train, y_train, X_val, y_val, X_test = load_data(LOC)
    mean = X_train.mean()
    std = X_train.std()

    train_imgs = transform_imgs(X_train, mean, std)
    train_labels = transform_labels(y_train)
    val_imgs = transform_imgs(X_val, mean, std)
    val_labels = transform_labels(y_val)
    weights = -torch.Tensor(np.array(np.unique(train_labels, return_counts=True))[1]/len(train_labels))

    train_data = TensorDataset(train_imgs, train_labels)
    val_data = TensorDataset(val_imgs,val_labels)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, weights
