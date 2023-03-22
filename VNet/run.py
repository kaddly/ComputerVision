import torch
from module import UNet3d
from train_utils import train_val
from utils import load_data

if __name__ == '__main__':
    lr = 1e-3
    batch_size = 2
    epochs = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    numclass = 5
    net = UNet3d(in_channels=1, out_channels=numclass)
    train_loader, val_loader = load_data(batch_size)
    train_val(net, numclass, train_loader, val_loader, epochs, device, lr)
