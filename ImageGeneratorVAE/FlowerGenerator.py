import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class FlowerVAE(pl.LightningModule):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True), # 32x32
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), # 16x16
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), # 8x8
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), # 4x4
            
            nn.Flatten() # 512 * 4 * 4 = 8192
        )
        
        self.fc_mu = nn.Linear(8192, latent_dim)
        self.fc_log_var = nn.Linear(8192, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 8192)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256), nn.ReLU(True), # 8x8
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128), nn.ReLU(True), # 16x16
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64), nn.ReLU(True), # 32x32
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), 
            nn.Tanh() # 64x64
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_log_var(x)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 512, 4, 4) 
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = (recon_loss + kld_loss) / x.size(0)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)