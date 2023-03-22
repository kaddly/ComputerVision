import os
import time
from tqdm import tqdm
from pathlib import Path
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from .losses import MutilDiceLoss
from .metric import dice_coeff, iou_coeff, multiclass_dice_coeff, multiclass_iou_coeff
from .visualization import save_images3d, plot_result
from .optimizer_utils import create_lr_scheduler


def initialize_weights(net):
    if isinstance(net, (nn.Conv3d, nn.Conv2d)):
        nn.init.kaiming_normal_(net.weight.data, nonlinearity='relu')
        if net.bias is not None:
            nn.init.constant_(net.bias.data, 0)
    elif isinstance(net, (nn.ConvTranspose3d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(net.weight.data, nonlinearity='relu')
        if net.bias is not None:
            nn.init.constant_(net.bias.data, 0)
    elif isinstance(net, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.constant_(net.weight.data, 1)
        if net.bias is not None:
            nn.init.constant_(net.bias.data, 0)
    elif isinstance(net, nn.Linear):
        nn.init.kaiming_uniform_(net.weight.data)
        nn.init.constant_(net.bias.data, 0)


def accuracy_function(numclass, accuracyname, input, target):
    if accuracyname == 'dice':
        if numclass == 1:
            return dice_coeff(input, target)
        else:
            return multiclass_dice_coeff(input, target)
    if accuracyname == 'iou':
        if numclass == 1:
            return iou_coeff(input, target)
        else:
            return multiclass_iou_coeff(input, target, numclass)


def train_val(model, numclass, train_loader, val_loader, epochs, device, lr=1e-3, showwind=[8, 12]):

    model_dir = './models'
    accuracyname = 'iou'
    showpixelvalue = 255.

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    MODEL_PATH = os.path.join(model_dir, "MutilUNet3d.pth")
    print("[INFO] training the network...")
    H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
    startTime = time.time()
    best_validation_dsc = 0.0

    model.to(device)
    model.apply(initialize_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs)
    # each class weight shape[0] is background
    alpha = torch.as_tensor([0.8, 1.2, 1.2, 1.2, 1.2]).contiguous().to(device)
    lossFunc = MutilDiceLoss(alpha)

    writer = SummaryWriter(log_dir=model_dir)

    for epoch in range(epochs):
        model.train()

        totalTrainLoss = []
        totalTrainAccu = []
        totalValidationLoss = []
        totalValidationAccu = []
        trainshow = True

        for batch in tqdm(train_loader):
            img, msk = [data.to(device) for data in batch]

            pred_logit, pred = model(img)
            loss = lossFunc(pred_logit, msk)
            accu = accuracy_function(numclass, accuracyname, pred, msk)

            if trainshow:
                savepath = model_dir + '/' + str(epoch + 1) + "_train_EPOCH_"
                save_images3d(torch.argmax(pred[0], 0), msk[0], showwind, savepath,
                              pixelvalue=showpixelvalue)
                trainshow = False

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # add the loss to the total training loss so far
            totalTrainLoss.append(loss)
            totalTrainAccu.append(accu)

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                img, msk = [data.to(device) for data in batch]

                pred_logit, pred = model(img)
                loss = lossFunc(pred_logit, msk)
                accu = accuracy_function(numclass, accuracyname, pred, msk)

                savepath = model_dir + '/' + str(epoch + 1) + "_Val_EPOCH_"

                save_images3d(torch.argmax(pred[0], 0), msk[0], showwind, savepath,
                              pixelvalue=showpixelvalue)
                totalValidationLoss.append(loss)
                totalValidationAccu.append(accu)
        avgTrainLoss = torch.mean(torch.stack(totalTrainLoss))
        avgValidationLoss = torch.mean(torch.stack(totalValidationLoss))
        avgTrainAccu = torch.mean(torch.stack(totalTrainAccu))
        avgValidationAccu = torch.mean(torch.stack(totalValidationAccu))

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["valdation_loss"].append(avgValidationLoss.cpu().detach().numpy())
        H["train_accuracy"].append(avgTrainAccu.cpu().detach().numpy())
        H["valdation_accuracy"].append(avgValidationAccu.cpu().detach().numpy())

        print("[INFO] EPOCH: {}/{}".format(epoch + 1, epochs))
        print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
            avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
        writer.add_scalar('Train/Loss', avgTrainLoss, epoch + 1)
        writer.add_scalar('Train/accu', avgTrainAccu, epoch + 1)
        writer.add_scalar('Valid/loss', avgValidationLoss, epoch + 1)
        writer.add_scalar('Valid/accu', avgValidationAccu, epoch + 1)
        writer.flush()

        if avgValidationAccu > best_validation_dsc:
            best_validation_dsc = avgValidationAccu
            torch.save(model.state_dict(), MODEL_PATH)
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
