import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from torchvision.io import decode_image



class LetterDS(Dataset):
    
    def __init__(self, root_dir, transform=None, sel='real'):
        csvf = os.path.join(root_dir,'literki', 'letters_labels_cleaned.csv')
        self.label_map = {'real': 0, 'fake': 1}
        self.finfo = pd.read_csv(csvf)
        # here is new
        #self.finfo = self.finfo.loc[self.finfo.letter.isin(['e','a', 'u', 't'])]
        self.finfo.reset_index(drop=True, inplace=True)
        # end new
        self.root_dir = root_dir
        self.transform = transform

        # use only available files
        for idx in range(len(self.finfo)):
            fn = self.get_filename(idx)
            in_disk = False
            if os.path.isfile(fn):
                in_disk = True
            self.finfo.loc[idx,'in_disk'] = in_disk
        self.finfo = self.finfo[self.finfo.in_disk]
        
            
        if sel != 'all':
            sel_label = self.label_map[sel]
            self.finfo = self.finfo[self.finfo.label==sel_label]
        self.finfo.reset_index(drop=True, inplace=True)
            
    def get_filename(self, idx):
        return os.path.join(
            self.root_dir,
            'literki',
            'letters_new',
            self.finfo.filename[idx]
        )
            
    def __len__(self):
        return len(self.finfo)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.get_filename(idx)
        image = Image.open(img_name)
        label = self.finfo.label[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
                
def get_dataset_mnist(data_dir, target_img_size):
    return datasets.MNIST(
        root=data_dir,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(target_img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])
    )

def weights_init_gan(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02) #(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.01) #(1.0, 0.02)
        model.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, z_dim, g_hidden, image_channel):
        super(Generator, self).__init__()        
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(z_dim, g_hidden * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_hidden * 8),
            nn.ReLU(inplace=False),
            # 1st hidden layer
            nn.ConvTranspose2d(g_hidden * 8, g_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden * 4),
            nn.ReLU(inplace=False),
            # 2nd hidden layer
            nn.ConvTranspose2d(g_hidden * 4, g_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden * 2),
            nn.ReLU(inplace=False),
            # 3rd hidden layer
            nn.ConvTranspose2d(g_hidden * 2, g_hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden),
            nn.ReLU(inplace=False),
            # output layer
            nn.ConvTranspose2d(g_hidden, image_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


    
class Discriminator(nn.Module):
    def __init__(self, d_hidden, image_channel):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(image_channel, d_hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # 2nd layer
            nn.Conv2d(d_hidden, d_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # 3rd layer
            nn.Conv2d(d_hidden * 2, d_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # 4th layer
            nn.Conv2d(d_hidden * 4, d_hidden * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 8),
            nn.LeakyReLU(0.2, inplace=False),
            # output layer
            nn.Conv2d(d_hidden * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.epsilon = 1e-10

    def __call__(self, sample):
        x = sample
        x = (x - self.mean)/(self.std+self.epsilon)
        return x    

class IGANMgr():
    # IGAN manager
    def __init__(self, conf):
        self.conf = conf
        self.model = {}
        self.optim = {}

    def save_generator(self, path='generator.pth'):
        torch.save(self.model['generator'].state_dict(), path)

    def save_discriminator(self, path='discriminator.pth'):
        torch.save(self.model['discriminator'].state_dict(), path)

    def set_comp_device(self):
        use_cuda = self.conf["cuda"] and torch.cuda.is_available()

        self.device = torch.device(
            "cuda" if use_cuda else "cpu"
        )

        if use_cuda:
            torch.cuda.manual_seed(self.conf["seed"])
            cudnn.benchmark = True

    def get_data_mnist(self):        
        self.dset = get_dataset_mnist(
            self.conf['data_path'],
            self.conf['x_dim']
        )
        
        self.dloader = DataLoader(
            self.dset,
            batch_size=self.conf['batch_size'],
            shuffle=True,
            num_workers=self.conf['num_workers']
        )

    def get_data_letters(self):
        target_size = (self.conf['x_dim'], self.conf['x_dim'])
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize((0.5,),(0.5,))
        ])
        ds = LetterDS(self.conf['data_path'],transform=transform, sel='all') 
        self.dset, self.dset_test = random_split(ds, [0.7, 0.3])
        
        n_samples = len(self.dset)
        batch_size = get_batch(self.conf['batch_size'], n_samples)
        print(f"desired batch size: {self.conf['batch_size']}, calculated batch size:{batch_size}")
        self.dloader = DataLoader(
            self.dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.conf['num_workers']
        )
        
    def view_samples(self, dloader): 
        side = 8 # image square side
        num_img = side*side        
        # bath_size bigger or equal than num_img
        real_batch = next(iter(dloader))
        plt.figure(figsize=(side,side))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(
                make_grid(
                    real_batch[0].to(self.device)[:num_img],
                    padding=2,
                    normalize=True).cpu(),
                (1,2,0)
            )
        )
        plt.draw()

    def make_generator(self):
        self.model['generator'] = Generator(
            self.conf['z_dim'],
            self.conf['g_hidden'],
            self.conf['image_channel']
        ).to(self.device)
        self.model['generator'].apply(weights_init_gan)

    def make_discriminator(self):
        self.model['discriminator'] = Discriminator(
            self.conf['d_hidden'],
            self.conf['image_channel']
        ).to(self.device)
        self.model['discriminator'].apply(weights_init_gan)

        
    def print_models(self):
        for model_type in self.model.keys():
            print(self.model[model_type])
            
    def set_loss(self):
        self.loss = nn.BCELoss()
        
    def set_optimizer(self):
        self.optim['generator'] = optim.Adam(
            self.model['generator'].parameters(),
            lr=self.conf['lr']['generator'],
            betas=(0.5, 0.999)
        )
        self.optim['discriminator'] = optim.Adam(
            self.model['discriminator'].parameters(),
            lr=self.conf['lr']['discriminator'],
            betas=(0.5, 0.999)
        )

    def create_latent_imgs(self):
        self.gimg = torch.randn(
            self.conf['batch_size'],
            self.conf['z_dim'],
            1,
            1,
            device=self.device
        )

    def viz_gen(self,index=-1):
        real_batch = next(iter(self.dloader))
        side = 8
        num_img = side*side
        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(
                make_grid(
                    real_batch[0].to(self.device)[:num_img],
                    padding=5,
                    normalize=True).cpu(),
                (1,2,0)
            )
        )

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        imgt = self.coll['img'][index].cpu()
        imgf = np.transpose(imgt.cpu(),(1,2,0))
        plt.imshow(imgf)
        plt.draw()

    def plot_loss(self):
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.title("Generator loss")
        plt.plot(self.coll['generator']['loss'])
        plt.subplot(1,2,2)
        plt.title("Discriminator loss")
        plt.plot(self.coll['discriminator']['loss'])
        plt.draw()



    def make_label_data(self, size, value=1):
        return torch.full(
            (size,),
            value,
            dtype=torch.float,
            device=self.device
        )

    def make_data_collectors(self):
        self.coll = {}
        self.coll['img'] = []
        self.coll['generator'] = {'loss':[]}
        self.coll['discriminator'] = {'loss':[]}        

    def train(self):
        iters = 0
        netD = self.model['discriminator']
        netG = self.model['generator']
        print("Starting Training Loop...")
        for epoch in range(self.conf['epoch_num']):
            for i, data in enumerate(self.dloader, 0):
                # (1) Update the discriminator with real data
                netD.zero_grad()
                # Format batch
                ## including fakes from database
                data_input = data[0].to(self.device)
                ## real_cpu = data[0].to(self.device)
                ## b_size = real_cpu.size(0)
                b_size = data_input.size(0)
                ## label = self.make_label_data(b_size)
                label = data[1].float().to(self.device)
                # data count by label
                # N_fake = label.sum()
                # N_real = label.size(0) - N_fake
                # print(label.size(0), N_real.cpu(), N_fake.cpu())
                                
                # Forward pass real batch through D
                ## output = netD(real_cpu).view(-1)
                output = netD(data_input).view(-1)
                # Calculate loss on all-real batch
                # errD_real = self.loss(output, label*self.conf['real_label'])
                ## Calculate loss on all-data batch (original real+fake) 
                errD_all =  self.loss(output, label) 
                # Calculate gradients for D in backward pass
                ## errD_real.backward()
                errD_all.backward()
                D_x = output.mean().item()

                # (2) Update the discriminator with fake data
                # Generate batch of latent vectors
                noise_size = (b_size, self.conf['z_dim'], 1, 1)
                noise = torch.randn(
                    size=noise_size,
                    device=self.device)
                # Generate fake image batch with G
                fake = netG(noise)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                # generate labels for generated fakes
                gen_label = self.make_label_data(b_size, self.conf['fake_label'])
                output = torch.clamp(output, 1e-10, 0.99)
                # N_gen_fake = b_size                
                errD_fake = self.loss(output, gen_label) #
                # Calculate the gradients for this batch, accumulated
                # (summed) with previous gradients                
                errD_fake.backward()
                
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                ## errD = errD_real + errD_fake
                errD = errD_all + errD_fake
                # Update D
                self.optim['discriminator'].step()

                # (3) Update the generator with fake data
                netG.zero_grad()
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                # here maybe mistake
                # errG = self.loss(output, label*self.conf['real_label'])
                gen_label = self.make_label_data(b_size, self.conf['real_label'])
                errG = self.loss(output, gen_label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optim['generator'].step()
                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.conf['epoch_num'], i, len(self.dloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later

                gloss = errG.item()
                self.coll['generator']['loss'].append(gloss)
                self.coll['discriminator']['loss'].append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.conf['epoch_num']-1) and (i == len(self.dloader)-1)):
                    with torch.no_grad():
                        fake = netG(self.gimg).detach().cpu()
                    N = fake.shape[0]
                    N = min(N, 96)
                    fake = fake[:N,...]
                    self.coll['img'].append(make_grid(fake, padding=2, normalize=True))
                iters += 1
            if (epoch + 1) % 5 == 0 or (epoch == self.conf['epoch_num'] - 1):
                gen_path = f'generator_epoch_{epoch + 1}.pth'
                disc_path = f'discriminator_epoch_{epoch + 1}_6.pth'
                self.save_generator(gen_path)
                print(f"Generator saved to {gen_path}")
                self.save_discriminator(disc_path)
                print(f"Discriminator saved to {disc_path}")

    def run(self, data_set):
        torch.cuda.empty_cache()
        self.set_comp_device()
        self.set_loss()        
        self.make_generator()
        self.make_discriminator()
        self.set_optimizer()
        if data_set == 'letters_new':
            self.get_data_letters()
        else:
            self.get_data_mnist()
            
        self.view_samples(self.dloader)
        
        self.print_models()
        self.create_latent_imgs()
        self.make_data_collectors()
        self.train()
        self.plot_loss()
        self.viz_gen()
        self.save_generator('generator.pth')
        print("Generator saved to generator.pth")
        self.save_discriminator('discriminator.pth')
        print("Discriminator saved to discriminator.pth")
        plt.show()
      
def get_batch(batch_size, n_samples, radius=10, verbose=False):
    '''Simple heuristic to equalize batch_size and last_batch size'''
    if n_samples%batch_size == 0:
        b = batch_size
    else:
        deltas = []
        for k in range(radius):
            deltas += [-k,k]
            diff = []
        for d in deltas:
            lb = n_samples%(batch_size + d)
            d_cal_last = abs(batch_size + d -lb)/batch_size
            d_des_cal =  abs(d)/batch_size
            goal = d_cal_last**2 + 0.5*d_des_cal**2
            diff.append([goal, batch_size, lb, d])
            diff.sort()
            b = batch_size + diff[0][-1]
        if verbose:
            for goal,b,lb,d in diff:
                print(
                    f'desired btch:{b:04d} '
                    f'cal. btch:{b+d:04d} '
                    f'last btch:{lb:04d} '
                    f'goal:{goal:02f}'
                )
    return b

if __name__ == '__main__':
    conf = {
        'cuda':True,
        # change data path with respect to your directory structure
        # images need to be located in data/PDFGAN/letters
        # csv_file need to be in  data/PDFGAN/letters_labels.csv
        'data_path':'.',
        'batch_size': 128, #2000, #128,
        'num_workers': 16,
        'image_channel': 1, # number of color channels of input image
        'z_dim': 100, #3  100 # dimension of the latent vector z (generator input)
        'g_hidden': 64, # size of feature maps in generator
        'x_dim': 64, # input image size (for resize)
        'd_hidden': 64, # size of feature maps in the discriminator
        'epoch_num': 100, #40 # number of epochs
        'real_label': 0,
        'fake_label': 1,
        'lr':{'generator':1e-3, 'discriminator':1e-3}, #2e-4, learning rates
        #'lr':{'generator':5e-3, 'discriminator':5e-3}, #2e-4, learning rates
        'seed':1354456345 # 1
    }
    ig = IGANMgr(conf)
    ig.run('letters_new') #'letters' run the pdf letters, 'mnist' run the mnist dat a

