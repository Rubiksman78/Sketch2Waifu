import torch
import torch.nn as nn
from models.losses import *

#Phase 1: X_fake1 = G(edge + mask * x_real)
#Phase 2: X_fake2 = G(edge + color domain)
#Phase 3: X_fake3 = G(edge + X_fake2)

def train_step(gen:nn.Module,disc:nn.Module,im_input,real_im,d_opt,g_opt,M,device):
    #Train D
    disc.zero_grad()
    real_output = disc(real_im)
    fake = gen(im_input)
    fake_output = disc(fake)
    real_label = 0.9*torch.ones_like(real_output)
    d_loss = disc_loss(real_output,real_label,fake_output)
    d_loss.backward()
    d_opt.step()

    #Train G
    gen.zero_grad()
    fake = gen(im_input)
    fake_output = disc(fake)
    real_label = torch.ones_like(fake_output)
    g_loss = gen_loss(fake,fake_output,real_label,real_im,M,device)
    g_loss.backward()
    g_opt.step()

    return d_loss,g_loss

def test_step(self,gen:nn.Module,disc:nn.Module,im_input,real_im,M,device):
    #Train D
    real_output = disc(real_im)
    fake = gen(im_input)
    fake_output = disc(fake)
    real_label = 0.9*torch.ones_like(real_output)
    d_loss = disc_loss(real_output,real_label,fake_output)

    #Train G
    fake = gen(im_input)
    fake_output = disc(fake)
    real_label = torch.ones_like(fake_output)
    g_loss = gen_loss(fake,fake_output,real_label,real_im,M,device)
    return d_loss, g_loss