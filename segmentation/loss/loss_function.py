import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class Dice_Loss_weight(nn.Module):

    def __init__(self):

        super(Dice_Loss_weight,self).__init__()

    def cal_dice_loss(self,input,target, weight,epsilon=1e-6):

        batchsize = input.size(0)
        input_label = input.view(batchsize, -1)
        target_label = target.view(batchsize, -1)
        self.intersect = torch.sum(input_label * target_label, 1)
        input_area = torch.sum(input_label, 1)
        target_area = torch.sum(target_label, 1)
        self.sum = input_area + target_area + 2 * epsilon
        batch_loss = 1 - 2 * self.intersect / self.sum
        batch_loss[target_area == 0] = 0
        batch_loss = batch_loss * weight
        loss = batch_loss.mean()

        return loss

    def update_weight(self,target):

        weight = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
        batchsize = target.size(0)
        input_wt = target[:, 0, :, :, :]
        input_wt_label = input_wt.view(batchsize, -1)
        input_wt_area = torch.sum(input_wt_label, 1)
        input_et = target[:, 2, :, :, :]
        input_et_label = input_et.view(batchsize, -1)
        input_et_label = torch.sum(input_et_label, 1)

        weight_1 = input_et_label / input_wt_area
        for i in range(8):
            if weight_1[i] > 0.1 and input_et_label[i] > 10000:
                weight[i] = 1
                continue
            elif weight_1[i] > 0.1 or input_et_label[i] > 10000:
                weight[i] = 0.8
                continue
            elif 0.1 >= weight_1[i] > 0.01 and 10000 >= input_et_label[i] > 1000:
                weight[i] = 0.5
                continue
            elif 0.1 >= weight_1[i] > 0.01 or 10000 >= input_et_label[i] > 1000:
                weight[i] = 0.3
                continue
            else:
                weight[i] = 0.1

        return  weight.cuda()

    def forward(self, input, target, save=True, epsilon=1e-6):

        loss = 0
        weight = torch.FloatTensor([1,1, 1,1,1,1,1,1])
        weight = weight.cuda()

        for i in range(3):
            if i == 2:
                weight = self.update_weight(target)
            loss +=self.cal_dice_loss(input[:, i, :, :, :],target[:, i, :, :, :],weight)
        loss = loss / 3

        return loss


class Dice_Loss_by_block(nn.Module):

    def __init__(self):
        super(Dice_Loss_by_block,self).__init__()

    def cal_dice_loss(self,input,target,epsilon=1e-6):

        batchsize = input.size(0)
        input_label = input.view(batchsize, -1)
        target_label = target.view(batchsize, -1)
        self.intersect = torch.sum(input_label * target_label, 1)
        input_area = torch.sum(input_label, 1)
        target_area = torch.sum(target_label, 1)
        self.sum = input_area + target_area + 2 * epsilon
        batch_loss = 1 - 2 * self.intersect / self.sum
        batch_loss[target_area == 0] = 0

        loss = torch.sum(batch_loss) / torch.sum(target_area != 0)
        # loss = batch_loss.mean()

        return loss

    def forward(self, input, target, block):

        loss = 0
        num = 0
        for i in range(10):
            block_tmp = block == i+1
            block_tmp = block_tmp.float()
            if block_tmp.max() != 0:
                num += 1
                input_tmp = input*block_tmp
                target_tmp = target*block_tmp

                loss += self.cal_dice_loss(input_tmp, target_tmp)
        loss = loss / num

        return loss,0


class Dice_Loss(nn.Module):

    def __init__(self, weight_target, weight_background):

        super(Dice_Loss,self).__init__()
        self.weight_target = weight_target
        self.weight_background = weight_background

    def forward(self, input, target, weight, epsilon=1e-6):
        
        weight[weight == 2] = self.weight_target
        weight[weight == 0] = self.weight_background 
        batchsize = input.size(0)
        input_label = input.view(batchsize, -1)
        target_label = target.view(batchsize, -1)
        weight_label = weight.view(batchsize, -1)

        self.intersect = torch.sum(input_label * target_label*weight_label, 1)
        input_area = torch.sum(input_label*weight_label, 1)
        target_area = torch.sum(target_label*weight_label, 1)
        self.sum = input_area + target_area + 2 * epsilon
        batch_loss = 1 - 2 * self.intersect / self.sum
        batch_loss[target_area == 0] = 0 
        loss = batch_loss.mean()
        
        return loss,batch_loss


class W_dice_Loss(nn.Module):

    def __init__(self):

        super(W_dice_Loss,self).__init__()

    def cal_dice_loss(self, input, target, save=True, epsilon=1e-6):

        batchsize = input.size(0)
        input_label = input.view(batchsize, -1)
        target_label = target.view(batchsize, -1)
        self.intersect = torch.sum(input_label * target_label, 1)
        input_area = torch.sum(input_label, 1)
        target_area = torch.sum(target_label, 1)
        self.sum = input_area + target_area + 2 * epsilon
        batch_loss = 1 - 2 * self.intersect / self.sum
        batch_loss[target_area == 0] = 0
        loss = batch_loss.mean()

        return loss

    def forward(self, input, target, save=True, epsilon=1e-6):
        loss = 0
        loss_log = []
        weight=[0,1]
        for i in range(2):
            tmp = self.cal_dice_loss(input[:, i, :, :, :],target[:, 0, :, :, :])
            loss +=tmp* weight[i]
            loss_log.append(tmp)

        # loss = loss / 2

        return loss,loss_log




# class Dice_ave_Loss(nn.Module):
#
#     def __init__(self):
#
#         super(Dice_ave_Loss,self).__init__()

class UnetVae_loss(nn.Module):

    def __init__(self):
        super(UnetVae_loss,self).__init__()

    def dice_loss(self,input, target):
        """soft dice loss"""
        eps = 1e-7
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

    def cal_dice_loss(self, input, target, epsilon=1e-6):
        batchsize = input.size(0)
        input_label = input.view(batchsize, -1)
        target_label = target.view(batchsize, -1)
        self.intersect = torch.sum(input_label * target_label, 1)
        input_area = torch.sum(input_label, 1)
        target_area = torch.sum(target_label, 1)
        self.sum = input_area + target_area + 2 * epsilon
        batch_loss = 1 - 2 * self.intersect / self.sum
        batch_loss[target_area == 0] = 0
        loss = batch_loss.mean()

        return loss

    def vae_loss(self,recon_x, x, mu, logvar):
        loss_dict = {}
        batchsize = recon_x.size(0)
        loss_kld = 0
        loss_recon = 0
        for i in range(batchsize):
            loss_kld +=   -0.5 * torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp())
            loss_recon += F.mse_loss(recon_x[i], x[i], reduction='mean')
        loss_dict['KLD'] = loss_kld / batchsize
        loss_dict['recon_loss'] =loss_recon / batchsize
        return loss_dict

        # loss_dict = {}
        # loss_dict['KLD'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # loss_dict['recon_loss'] = F.mse_loss(recon_x, x, reduction='mean')
        #
        # return loss_dict

    # def forward(self, batch_pred, batch_x, batch_y, vout, mu, logvar):
    #     loss_dict = {}
    #     loss_dict['dice_loss'] =self.cal_dice_loss(batch_pred, batch_y)
    #     loss_dict.update(self.vae_loss(vout, batch_x, mu, logvar))
    #     weight = 0.1
    #     loss_dict['loss'] = loss_dict['dice_loss']  + weight * loss_dict['recon_loss'] + weight * loss_dict['KLD']
    #     return loss_dict

    def forward(self,batch_pred, batch_x, batch_y, vout, mu, logvar):

        loss_dict = {}
        loss_dict['wt_loss'] = self.cal_dice_loss(batch_pred[:, 0], batch_y[:, 0])  # whole tumor
        loss_dict['tc_loss'] = self.cal_dice_loss(batch_pred[:, 1], batch_y[:, 1])  # tumore core
        loss_dict['et_loss'] = self.cal_dice_loss(batch_pred[:, 2], batch_y[:, 2])
         # enhance tumor
        loss_dict.update(self.vae_loss(vout, batch_x, mu, logvar))
        weight = 0.1
        loss_dict['loss'] = loss_dict['wt_loss'] + loss_dict['tc_loss'] + loss_dict['et_loss'] + \
                             weight * loss_dict['recon_loss'] + weight * loss_dict['KLD']

        return loss_dict

    def get_losses(self,cfg):
        losses = {}
        losses['vae'] = self.vae_loss
        losses['dice'] = self.dice_loss
        losses['dice_vae'] = unet_vae_loss

        return losses