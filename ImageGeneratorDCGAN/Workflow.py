import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

from FlowerGAN import FlowerGAN 

def main():
    IMG_SIZE = 64
    BATCH_SIZE = 32
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mentesi_mappa = f"eredmenyek/GAN_futtatas_{timestamp}"
    os.makedirs(mentesi_mappa, exist_ok=True) 
    print(f"--- A GAN eredmények ide fognak mentődni: {mentesi_mappa} ---")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root="./flower_photos", transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = FlowerGAN()
    logger = CSVLogger("logs", name="flower_gan") 
    
    trainer = pl.Trainer(
        max_epochs=104,
        accelerator="cpu", 
        devices=1,
        logger=logger
    )
    print("--- Elindul a GAN tanítása (Generátor vs Diszkriminátor)... ---")
    trainer.fit(model, train_loader)

    print(f"--- GAN eredmények mentése ide: {mentesi_mappa} ... ---")
    model.eval()
    
    # A) Tanítási görbe: A két AI harca
    try:
        metrics = pd.read_csv(f"{logger.log_dir}/metrics.csv")
        metrics = metrics.groupby('epoch').mean().reset_index()
        
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['epoch'], metrics['g_loss'], label='Generátor (Hamisító) Hibája', color='blue', linewidth=2)
        plt.plot(metrics['epoch'], metrics['d_loss'], label='Diszkriminátor (Rendőr) Hibája', color='red', linewidth=2)
        plt.title("GAN Tanulási Görbe: A Két Hálózat Harca")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{mentesi_mappa}/01_GAN_tanitasi_gorbe.png")
        plt.close()
    except Exception as e:
        print(f"Nem sikerült a loss görbét megrajzolni: {e}")

    with torch.no_grad():
        random_noise = torch.randn(8, 100) 
        new_flowers = model(random_noise)
        
        plt.figure(figsize=(16, 8))
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.title(f"AI Virág #{i+1}")
            plt.imshow((new_flowers[i].permute(1, 2, 0).numpy() + 1) / 2)
            plt.axis('off')
            
        plt.suptitle("DCGAN által generált ÚJ virágok", fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{mentesi_mappa}/02_GAN_generalt_viragok.png")
        plt.close()
        
    print(f"--- KÉSZ! Nézd meg a generált virágokat itt: {mentesi_mappa} ---")

if __name__ == "__main__":
    main()