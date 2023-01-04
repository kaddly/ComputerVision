import os
import torch
from matplotlib import pyplot as plt
from train import train
from model import UNet
from utils import Diffusion


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = os.path.join(os.path.abspath('.'), 'data', 'Chinese-Landscape-Painting-Dataset')
    args.device = "cuda"
    args.lr = 3e-4
    args.is_resent_train = True
    train(args)


if __name__ == '__main__':
    # launch()
    device = "cuda"
    model = UNet(device=device).to(device)
    ckpt = torch.load("./models/DDPM_Unconditional/ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, 1)
    print(x.shape)
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in x.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
