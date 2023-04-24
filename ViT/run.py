import argparse
from utils import setup_logging, load_flower_data
from module import vit_base_patch16_224
from train import train_eval


def set_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='ViT')

    args = parser.parse_args()
    setup_logging(args.run_name)


if __name__ == '__main__':
    pass
