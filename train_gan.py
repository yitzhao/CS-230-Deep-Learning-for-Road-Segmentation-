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
from models.gan_pytorch import build_dc_classifier, build_dc_generator, initialize_weights, get_optimizer, discriminator_loss, generator_loss
from models.dcgan import *
writer = SummaryWriter()

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
    # Tensor =  torch.FloatTensor
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # adversarial_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    dice_loss = DiceLoss()

    # Initialize generator and discriminator
    g_path = 'checkpoint/checkpoint_gan_unet_model/model_epoch_51_generator.pth'
    generator = Generator()
    generator.load_state_dict(torch.load(g_path))
    generator = generator.to('cuda')
    generator.eval()

    d_path = 'checkpoint/checkpoint_gan_unet_model/model_epoch_51_discriminator.pth'
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(d_path))
    discriminator = discriminator.to('cuda')
    discriminator.eval()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        mae_loss.cuda()
        dice_loss.cuda()
    
    # Initialize weights
    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=args.lr)

    torch.manual_seed(args.seed)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if cuda:
        torch.cuda.manual_seed(args.seed)

    print('===> Loading datasets')

    training_data_loader, valid_data_loader, testing_data_loader = load_datasets()
    print('===> Building GAN')   

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

            if cuda:
                input = input.cuda()
                target = target.cuda()

            valid = Variable(Tensor(target.shape[0], 1).fill_(1.0))
            # valid = Variable(Tensor(target.shape[0], 1, 32, 32).fill_(1.0), requires_grad=False)
            gen_imgs = generator(input)
            g_gan_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_dice_loss = dice_loss(gen_imgs, target)

            g_loss = g_gan_loss + 10*g_dice_loss
            prediction = gen_imgs.detach()

            totalloss += g_loss.item()

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
        writer.add_scalar("Generator Loss/validation", totalloss/len(valid_data_loader), (epoch-1)*270)
        writer.add_scalar("IOU/validation", totaliou/len(valid_data_loader), (epoch-1)*270)
        print("===> Avg. Validation Generation loss: {:.4f} dB".format(totalloss / len(valid_data_loader)))
        print("===> Avg. Validation iou: {:.4f} dB".format(totaliou / len(valid_data_loader)))



    def train(epoch):
        epoch_loss = 0
        totaliou = 0

        for iteration, batch in enumerate(training_data_loader, 1):
            input = Variable(batch[0],volatile=True)
            input = torch.permute(input, (0, 3, 1, 2))
            input = input.type(torch.float32)
            
            target = Variable(batch[1], volatile = True)
            target = torch.permute(target, (0, 3, 1, 2))
            target = target[:,1:2,:,:]
            target = target.type(torch.float32)
            if cuda:
                input = input.to('cuda')
                target = target.to('cuda')

            # Adversarial ground truths
            valid = Variable(Tensor(target.shape[0], 1).fill_(1.0))
            fake = Variable(Tensor(target.shape[0], 1).fill_(0.0))
            # valid = Variable(Tensor(target.shape[0], 1, 32, 32).fill_(1.0))
            # fake = Variable(Tensor(target.shape[0], 1, 32, 32).fill_(0.0))

            # Configure input
            real_imgs = Variable(target.type(Tensor))
            
            # Generate a batch of images
            gen_imgs = generator(input)

            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            
            optimizer_G.zero_grad()
            # Loss measures generator's ability to fool the discriminator
            g_gan_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_dice_loss = dice_loss(gen_imgs, target)

            g_loss = g_gan_loss + 10 * g_dice_loss
            epoch_loss += g_loss.item()

            g_loss.backward()
            optimizer_G.step()

            output = gen_imgs.detach()
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
                writer.add_scalar("g_Loss/train", g_loss, (epoch-1)*270+iteration)
                writer.add_scalar("d_loss/train", d_loss, (epoch-1)*270+iteration)
                print("===> Epoch[{}]({}/{}): g_loss: {:.4f}".format(epoch, iteration, len(training_data_loader), g_loss.item()))
                print("===> Epoch[{}]({}/{}): d_loss: {:.4f}".format(epoch, iteration, len(training_data_loader), d_loss.item()))
            
        # imgout = input.data/2 +1
        # torchvision.utils.save_image(imgout,"/cs230_project/checkpoint/epch_"+str(epoch)+'.jpg')
        print("===> Epoch {} Complete: Avg. g_loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
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

            if cuda:
                input = input.cuda()
                target = target.cuda()
            
            valid = Variable(Tensor(target.shape[0], 1).fill_(1.0))
            # valid = Variable(Tensor(target.shape[0], 1, 32, 32).fill_(1.0), requires_grad=False)
            gen_imgs = generator(input)
            g_gan_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_dice_loss = dice_loss(gen_imgs, target)
        
            g_loss = g_gan_loss + 10*g_dice_loss
            prediction = gen_imgs.detach()

            totalloss += g_loss.item()

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
        writer.add_scalar("Generator Loss/validation", totalloss, (epoch-1)*270)
        writer.add_scalar("IOU/validation", totaliou, (epoch-1)*270)
        print("===> Avg. test generator loss: {:.4f} dB".format(totalloss / len(testing_data_loader)))
        print("===> Avg. test iou: {:.4f} dB".format(totaliou / len(testing_data_loader)))


    def checkpoint(epoch):
        g_model_out_path = "checkpoint/checkpoint_gan_unet_model/model_epoch_{}_generator.pth".format(epoch)
        torch.save(generator.state_dict(), g_model_out_path)
        d_model_out_path = "checkpoint/checkpoint_gan_unet_model/model_epoch_{}_discriminator.pth".format(epoch)
        torch.save(discriminator.state_dict(), d_model_out_path)
        print("Checkpoint saved to {}".format(g_model_out_path))
    
    # for epoch in range(1+args.current_trained_epochs, 1+args.nEpochs + 1):
    #     train(epoch)
    #     checkpoint(epoch)
    #     validation(epoch)
    # test()

    writer.flush()
    writer.close()

    def visualize_trained_model(path, testing_data_loader):
        totalloss = 0
        totaliou = 0
        for batch in testing_data_loader:
            input = Variable(batch[0],volatile=True)
            input = torch.permute(input, (0, 3, 1, 2))
            input = input.type(torch.float32)
            # target = Variable(batch[1][:, :,
            #                 target_gap:target_gap + target_size,
            #                 target_gap:target_gap + target_size],
            #                 volatile=True)
            target = Variable(batch[1], volatile = True)
            target = torch.permute(target, (0, 3, 1, 2))
            target = target[:,1:2,:,:]
            target = target.type(torch.float32)
            #target =target.long().squeeze(1)
            if torch.cuda.is_available():
                input = input.to('cuda')
                target = target.to('cuda')
            # unet = UNet()
            generator = Generator()
            generator.load_state_dict(torch.load(path))
            generator = generator.to('cuda')
            generator.eval()
            # optimizer.zero_grad()
            gen_imgs = generator(input)

            prediction = gen_imgs.view(1, 1, 256, 256)
            prediction = prediction.reshape((256, 256))
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
            prediction = prediction.cpu().detach().numpy()
            
            target = target.cpu().detach().numpy().reshape((256,256))
            # input = torch.permute(input, (0, 2, 3, 1))
            # input = input.cpu().reshape(256,256,3).type(torch.int8)

            visualize(target = target, prediction = prediction)

    path = 'checkpoint/checkpoint_gan_unet_model/model_epoch_52_generator.pth'
    training_data_loader, valid_data_loader, testing_data_loader = load_datasets()
    visualize_trained_model(path, testing_data_loader)

if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("--cuda", default=True)
    p.add("--batchSize", default=4,type=int)
    p.add("--testBatchSize", default=4,type=int)
    p.add("--lr", default=0.001, type=float)
    # p.add("--gamma", default=0.9, type=float)
    # p.add("--period", default=3,type=int)
    p.add("--nEpochs", default=70,type=int)
    p.add("--threads", default=5,type=int)
    p.add("--seed", default=123,type=int)
    p.add("--size", default=428,type=int)
    p.add("--remsize", default=20,type=int)
    p.add("--colordim", default=3,type=int)
    p.add("--Load_weight", default=False)
    p.add("--weight_path")
    p.add("--current_trained_epochs", default=51, type=int)
    # p.add("--mode", Required=True)
    args = p.parse_args()
    main(args)
