import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import RandomSampler, DataLoader, Subset
from preprocess.dataset import Dataset
from models.generator import Generator
from models.discriminator import Discriminator
from torch.utils.data import DataLoader
from models.gan import train_step
from config import DEFAULT_CONFIG
from preprocess.color_domain import random_mask

#Load config
config = DEFAULT_CONFIG
num_samples,n_channels,lr,n_epochs,batch_size = config['NUM_SAMPLES']\
    ,config['NUM_CHANNELS'],config['LR'],config['N_EPOCHS'],config['BATCH_SIZE']

#Load dataset
flist = "../anime_face"
dataset = Dataset(config,flist,training=True)
sample_ds = Subset(dataset,np.arange(num_samples))
sampler = RandomSampler(sample_ds)

#Hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

def train(n_epochs,sample_ds,sampler,gen,disc,g_opt,d_opt,device,batch_size,checkpoint=6,mode=0):
    gen.load_state_dict(torch.load('weights/gen1.pt'))
    disc.load_state_dict(torch.load('weights/disc1.pt'))
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
            torch.save(gen.state_dict(),f"weights/gen{mode}.pt")
            torch.save(disc.state_dict(),f"weights/disc{mode}.pt")
            generate_image(gen,x,mode,epoch+checkpoint)
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
            generate_image(gen,x,mode,epoch+checkpoint)
        if mode == 2:
            #Infer previous mode on dataset and train on edge + infered images
            pass

#train(n_epochs,sample_ds,sampler,gen,disc,g_opt,d_opt,device,batch_size,mode=1)

def test(path):
    test_ds = Dataset(config,path,training=False)
    imgs,_,edges,cmap = next(iter(DataLoader(test_ds,batch_size=1)))
    gen.load_state_dict(torch.load('weights/gen1.pt'))
    cmap = cmap[0].squeeze().to(device)
    edge = edges[0].to(device)
    res = torch.cat((edge,cmap),dim=0)
    res = torch.unsqueeze(res,0)
    fake = gen(res).detach().cpu()
    im = fake.numpy().squeeze()
    im = im.transpose(1,2,0)
    im = im.clip(0,1)
    plt.axis('off')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cmap = cv2.cvtColor(cmap.detach().cpu().numpy().squeeze().transpose(1,2,0), cv2.COLOR_BGR2RGB)
    plt.subplot(131), plt.imshow(im), plt.title('Result')
    plt.subplot(132), plt.imshow(cmap), plt.title('Colormap')
    plt.subplot(133), plt.imshow(edge.detach().cpu().numpy().transpose(1,2,0)), plt.title('Edges')
    plt.show()

#test("test_img/3.png")