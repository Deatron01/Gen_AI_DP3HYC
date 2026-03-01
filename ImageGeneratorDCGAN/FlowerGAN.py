import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class FlowerGAN(pl.LightningModule):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        self.latent_dim = latent_dim

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(512), nn.ReLU(True), # 4x4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(256), nn.ReLU(True), # 8x8
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(128), nn.ReLU(True), # 16x16
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(64), nn.ReLU(True), # 32x32
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), 
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True), # 32x32
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), # 16x16
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), # 8x8
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), # 4x4
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, z):

        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        opt_g, opt_d = self.optimizers()

        valid = torch.ones(imgs.size(0), 1, device=self.device)
        fake = torch.zeros(imgs.size(0), 1, device=self.device)

        z = torch.randn(imgs.shape[0], self.latent_dim, device=self.device)
        gen_imgs = self.forward(z)

        real_loss = F.binary_cross_entropy(self.discriminator(imgs).view(-1, 1), valid)
        fake_loss = F.binary_cross_entropy(self.discriminator(gen_imgs.detach()).view(-1, 1), fake)
        
        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        g_loss = F.binary_cross_entropy(self.discriminator(gen_imgs).view(-1, 1), valid)

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("g_loss", g_loss, prog_bar=True)
        self.log("d_loss", d_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0.999
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []