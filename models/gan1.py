import torch
import torch.nn as nn
from losses import *

#Phase 1: X_fake = G(edge + mask * x_real)

def train_step(gen:nn.Module,disc:nn.Module,im_input,real_im,device,d_opt,g_opt,M):
    #Train D
    disc.zero_grad()
    real_output = disc(real_im).view(-1)
    fake = gen(im_input)
    fake_output = disc(fake).view(-1)
    real_label = torch.ones_like(real_output)
    d_loss = disc_loss(real_output,real_label,fake_output)
    d_loss.backward()
    d_opt.step()

    #Train G
    gen.zero_grad()
    fake = gen(im_input)
    fake_output = disc(fake).view(-1)
    real_label = torch.ones_like(fake_output)
    g_loss = gen_loss(fake,fake_output,real_label,real_im,M)
    g_loss.backward()
    g_opt.step()

    return d_loss,g_loss