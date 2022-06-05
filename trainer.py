import torch
import torch.nn as nn
from preprocess.dataset import Dataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.losses import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.gan import train_step
from config import config
from preprocess.color_domain import random_mask
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import cv2
#Load dataset
config = config
flist = "../anime_face"
dataset = Dataset(config,flist,training=True)
num_samples = 500
sample_ds = Subset(dataset,np.arange(num_samples))
sampler = RandomSampler(sample_ds)

#Hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
n_channels = 4
lr = 1e-3
n_epochs = 100
batch_size = 4
gen = Generator(n_channels).to(device)
disc = Discriminator(n_channels-1,64).to(device)
g_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
d_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

#This function generate an image and plot it
def generate_image(gen,imgs,mode,epoch):
    plt.figure(figsize=(10,10))
    for i in range(imgs.shape[0]):
        plt.subplot(1,imgs.shape[0],i+1)
        img = torch.unsqueeze(imgs[i],0)
        im = gen(img).detach().cpu()
        im = im.numpy().squeeze()
        im = im.transpose(1,2,0)
        im = im.clip(0,1)
        plt.axis('off')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im)
    plt.savefig(f'results/test{mode}{epoch}.png')
    plt.close()

def train(n_epochs,sample_ds,sampler,gen,disc,g_opt,d_opt,device,batch_size,mode=0):
    gen.load_state_dict(torch.load('weights/gen03.pt'))
    disc.load_state_dict(torch.load('weights/disc03.pt'))
    for epoch in range(n_epochs):
        progress_bar = tqdm(DataLoader(sample_ds,sampler=sampler,batch_size=batch_size))
        if mode == 0:
            for _,(img,img_gray,edge,color_domain) in enumerate(progress_bar):
                img,img_gray,edge,color_domain = img.to(device),img_gray.to(device),edge.to(device),color_domain.to(device)
                M = torch.from_numpy(random_mask(img)).to(device)
                x = torch.cat((edge,M*img),dim=1)
                x = x.float().to(device)
                y = img.to(device)
                d_loss,g_loss = train_step(gen,disc,x,y,d_opt,g_opt,M,device)
                progress_bar.set_description(f"\r[{epoch}/{n_epochs}] d_loss: {d_loss:.3f}, g_loss: {g_loss:.3f}")
            torch.save(gen.state_dict(),f"weights/gen{mode}{epoch}.pt")
            torch.save(disc.state_dict(),f"weights/disc{mode}{epoch}.pt")
            generate_image(gen,x,mode,epoch)
        if mode == 1:
            for _,(img,img_gray,edge,color_domain) in enumerate(progress_bar):
                img,img_gray,edge,color_domain = img.to(device),img_gray.to(device),edge.to(device),color_domain.to(device)
                M = torch.ones_like(img).to(device)
                x = torch.cat((edge,color_domain),dim=1)
                x = x.float().to(device)
                y = img.to(device)
                d_loss,g_loss = train_step(gen,disc,x,y,d_opt,g_opt,M,device)
                progress_bar.set_description(f"\r[{epoch}/{n_epochs}] d_loss: {d_loss:.3f}, g_loss: {g_loss:.3f}")
            torch.save(gen.state_dict(),f"weights/gen{mode}.pt")
            torch.save(disc.state_dict(),f"weights/disc{mode}.pt")
            generate_image(gen,x,mode,epoch)

train(n_epochs,sample_ds,sampler,gen,disc,g_opt,d_opt,device,batch_size,mode=1)

def test():
    test_ds = Dataset(config,"D:/Datasets",training=False)
    imgs,_,edges,_ = next(iter(DataLoader(test_ds,batch_size=2)))
    gen.load_state_dict(torch.load('weights/gen03.pt'))
    img = imgs[0].squeeze().to(device)
    edge = edges[0].to(device)
    res = torch.cat((img,edge),dim=0)
    res = torch.unsqueeze(res,0)
    fake = gen(res).detach().cpu()
    im = fake.numpy().squeeze()
    im = im.transpose(1,2,0)
    im = im.clip(0,1)
    plt.axis('off')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.show()
