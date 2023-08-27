#train.py
from __future__ import print_function
from email.policy import default
from math import log10
from typing_extensions import Required
import numpy as np
import random
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import segmentation_models_pytorch as smp
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.unet import UNet
from torchmetrics import JaccardIndex
# from BSDDataLoader import get_training_set,get_test_set
# from Dataloader import RoadsDataset
from utils.data_load import *
from utils.data_load import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def map01(tensor,eps=1e-5):
    #input/output:tensor
    max = np.max(tensor.numpy(), axis=(1,2,3), keepdims=True)
    min = np.min(tensor.numpy(), axis=(1,2,3), keepdims=True)
    if (max-min).any():
        return torch.from_numpy( (tensor.numpy() - min) / (max-min + eps) )
    else:
        return torch.from_numpy( (tensor.numpy() - min) / (max-min) )



def sizeIsValid(size):
    for i in range(4):
        size -= 4
        if size%2:
            return 0
        else:
            size /= 2
    for i in range(4):
        size -= 4
        size *= 2
    return size-4



class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        # bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return (1 - dice_coef)

def main(args):
    target_size = sizeIsValid(args.size)
    print("outputsize is: "+str(target_size))
    if not target_size:
        raise  Exception("input size invalid")
    target_gap = (args.size - target_size)//2
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    print('===> Loading datasets')

    training_data_loader, valid_data_loader, testing_data_loader = load_datasets()
    # Get train and val data loaders
    # training_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    # valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    # testing_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('===> Building unet')
    unet = UNet(args.colordim)


    # criterion = smp.losses.DiceLoss(mode = 'binary')
    # criterion = nn.MSELoss()
    class_weights=torch.tensor([9],dtype=torch.float)
    if cuda:
        class_weights = class_weights.to('cuda')
        
    # criterion = nn.BCEWithLogitsLoss(pos_weight = class_weights)
    criterion = DiceLoss()
    if cuda:
        unet = unet.cuda()
        criterion = criterion.cuda()

 
    if args.Load_weight:
        unet.load_state_dict(torch.load(args.weight_path))
        print("===> Pretrained Weights loaded")
    optimizer = optim.SGD(unet.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    def validation(epoch):
        totalloss = 0
        totaliou = 0

        for batch in valid_data_loader:
            input = Variable(batch[0],volatile=True)
            input = torch.permute(input, (0, 3, 1, 2))
            input = input.type(torch.float32)

            target = Variable(batch[1], volatile = True)
            target = torch.permute(target, (0, 3, 1, 2))
            target = target[:,1:2,:,:]
            target = target.type(torch.float32)
            #target =target.long().squeeze(1)
            if cuda:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            prediction = unet(input)
            
            loss = criterion(prediction, target)
            totalloss += loss.data

            jaccard = JaccardIndex(num_classes=2)
            if cuda:
                jaccard = jaccard.to('cuda')
            target = target.type(torch.int8)
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            prediction = prediction.type(torch.int8)
            if cuda:
                prediction = prediction.to('cuda')
                target = target.to('cuda')
            iou = jaccard(prediction, target)
            totaliou += iou.data
        writer.add_scalar("Loss/validation", totalloss/ len(valid_data_loader), (epoch-1)*270)
        writer.add_scalar("IOU/validation", totaliou/ len(valid_data_loader), (epoch-1)*270)
        print("===> Avg. Validation loss: {:.4f} dB".format(totalloss / len(valid_data_loader)))
        print("===> Avg. Validation iou: {:.4f} dB".format(totaliou / len(valid_data_loader)))



    def train(epoch):
        epoch_loss = 0
        totaliou = 0

        for iteration, batch in enumerate(training_data_loader, 1):
            input = Variable(batch[0],volatile=True)
            # print(input.size())
            input = torch.permute(input, (0, 3, 1, 2))
            # print(input.size())
            input = input.type(torch.float32)
            
            target = Variable(batch[1], volatile = True)
            target = torch.permute(target, (0, 3, 1, 2))
            target = target[:,1:2,:,:]
            target = target.type(torch.float32)
            #target =target.squeeze(1)
            #print(target.data.size())
            if cuda:
                input = input.to('cuda')
                target = target.to('cuda')

            # input_im = torch.permute(input, (0, 2, 3, 1))
            # input_im = input_im[0,:,:,:].cpu().detach().numpy()
            # input_im = input_im.reshape((256, 256, 3))
            # plt.imshow(input_im, aspect='auto')
            # plt.show()

            output = unet(input)
            #print(input.data.size())
            # print(input.size())
            # print(target.size())
            loss = criterion(output, target)
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()

            jaccard = JaccardIndex(num_classes=2)
            if cuda:
                jaccard = jaccard.to('cuda')
            target = target.type(torch.int8)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            output = output.type(torch.int8)

            if cuda:
                output = output.to('cuda')
                target = target.to('cuda')
            iou = jaccard(output, target)
            totaliou += iou.data


            if iteration%10 == 0:
                writer.add_scalar("Loss/train", loss, (epoch-1)*270+iteration)
                writer.add_scalar("IOU/train", iou, (epoch-1)*270+iteration)
                print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data))
            
        # imgout = input.data/2 +1
        # torchvision.utils.save_image(imgout,"/cs230_project/checkpoint/epch_"+str(epoch)+'.jpg')
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
        print("===> Epoch {} Complete: Avg. iou: {:.4f}".format(epoch, totaliou / len(training_data_loader)))

    def test():
        totalloss = 0
        totaliou = 0
        for batch in testing_data_loader:
            input = Variable(batch[0],volatile=True)
            input = torch.permute(input, (0, 3, 1, 2))
            input = input.type(torch.float32)

            target = Variable(batch[1], volatile = True)
            target = torch.permute(target, (0, 3, 1, 2))
            target = target[:,1:2,:,:]
            target = target.type(torch.float32)
            #target =target.long().squeeze(1)
            if cuda:
                input = input.to('cuda')
                target = target.to('cuda')
            optimizer.zero_grad()
            prediction = unet(input)
            
            loss = criterion(prediction, target)
            totalloss += loss.data

            jaccard = JaccardIndex(num_classes=2)
            if cuda:
                jaccard = jaccard.to('cuda')
            target = target.type(torch.int8)
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            prediction = prediction.type(torch.int8)
            if cuda:
                prediction = prediction.to('cuda')
                target = target.to('cuda')
            iou = jaccard(prediction, target)
            totaliou += iou.data
            
        print("===> Avg. test loss: {:.4f} dB".format(totalloss / len(testing_data_loader)))
        print("===> Avg. test iou: {:.4f} dB".format(totaliou / len(testing_data_loader)))


    def checkpoint(epoch):
        model_out_path = "checkpoint/checkpoint_unet_model/model_epoch_{}.pth".format(epoch)
        torch.save(unet.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
    
    for epoch in range(1+args.current_trained_epochs, 1+args.nEpochs + 1):
        train(epoch)
        checkpoint(epoch)
        lr_scheduler.step()
            
        if epoch%5 == 0:
            validation(epoch)
    test()
    writer.flush()
    writer.close()
        # checkpoint(epoch)

    # def visualize_trained_model(path, testing_data_loader):
    #     totalloss = 0
    #     totaliou = 0
    #     for batch in testing_data_loader:
    #         input = Variable(batch[0],volatile=True)
    #         input = torch.permute(input, (0, 3, 1, 2))
    #         input = input.type(torch.float32)
    #         # target = Variable(batch[1][:, :,
    #         #                 target_gap:target_gap + target_size,
    #         #                 target_gap:target_gap + target_size],
    #         #                 volatile=True)
    #         target = Variable(batch[1], volatile = True)
    #         target = torch.permute(target, (0, 3, 1, 2))
    #         target = target[:,1:2,:,:]
    #         target = target.type(torch.float32)
    #         #target =target.long().squeeze(1)
    #         if torch.cuda.is_available():
    #             input = input.to('cuda')
    #             target = target.to('cuda')
    #         unet = UNet()
    #         unet.load_state_dict(torch.load(path))
    #         unet = unet.to('cuda')
    #         unet.eval()
    #         # optimizer.zero_grad()
    #         prediction = unet(input)
    #         prediction = prediction.data[0,:,:,:].cpu().detach().numpy()
    #         prediction = prediction.reshape((256,256))
            
    #         # target = target.cpu().detach().numpy().reshape((256,256))

    #         plt.imshow(prediction)
    #         plt.show()

    # path = 'checkpoint/checkpoint_dice/model_epoch_10.pth'
    # training_data_loader, valid_data_loader, testing_data_loader = load_datasets()
    # visualize_trained_model(path, testing_data_loader)

    # if args.mode == "Train":
    #     print('===> Training unet')
    #     train(args.nEpochs)
    # elif args.mode == "Test":
    #     print('===> Testing unet')
    #     test()
    
if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("--cuda", default=True)
    p.add("--batchSize", default=4,type=int)
    p.add("--testBatchSize", default=4,type=int)
    p.add("--lr", default=0.001, type=float)
    # p.add("--gamma", default=0.9, type=float)
    # p.add("--period", default=3,type=int)
    p.add("--nEpochs", default=50,type=int)
    p.add("--threads", default=5,type=int)
    p.add("--seed", default=123,type=int)
    p.add("--size", default=428,type=int)
    p.add("--remsize", default=20,type=int)
    p.add("--colordim", default=3,type=int)
    p.add("--Load_weight", default=False)
    p.add("--weight_path")
    p.add("--current_trained_epochs", default=0, type=int)
    # p.add("--mode", Required=True)
    args = p.parse_args()
    main(args)