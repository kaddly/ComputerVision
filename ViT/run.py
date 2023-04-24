import os
import argparse
from utils import setup_logging, load_flower_data
from module import vit_base_patch16_224_in21k
from train import train_eval


def set_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='ViT')
    parser.add_argument('--data_path', type=str, default=os.path.join('data', 'flower_photos'))
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze_layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()
    setup_logging(args.run_name)
    return args


if __name__ == '__main__':
    args = set_param()
    train_loader, val_loader = load_flower_data(args)
    model = vit_base_patch16_224_in21k(args.num_classes, has_logits=False)
    train_eval(model, train_loader, val_loader, args)
