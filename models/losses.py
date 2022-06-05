import torch
import torch.nn as nn
import torchvision

#This function refers to the total number of non-zero pixels in the image tensor
def F_sum(img):
    return torch.sum(img.view(-1))

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        feat_loss,style_loss=0,0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                feat_loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                style_loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return feat_loss,style_loss

def disc_loss(real_output,real_label,fake_output):
    real_loss = torch.mean((real_output - real_label)**2)
    fake_loss = torch.mean((fake_output)**2)
    return 0.5*real_loss + 0.5*fake_loss

def gen_loss(fake_im,fake_output,real_label,real_im,M,device,alpha=1,beta=.01,gamma=1,delta=150):
    loss_adv = 0.5*torch.mean((fake_output - real_label)**2)
    loss_per_pixel = F_sum(real_im)/F_sum(M) * nn.L1Loss()(fake_im,M * real_im)
    feat_loss,style_loss = VGGPerceptualLoss().to(device)(fake_im,real_im)
    return beta*loss_adv + alpha*loss_per_pixel + gamma*feat_loss + delta*style_loss

