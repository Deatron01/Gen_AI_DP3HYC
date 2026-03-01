import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime # <-- Ezt importáltuk az időbélyeghez

# Figyelj az importra: megváltozott a háló neve!
from FlowerGenerator import FlowerVAE 

def main():
    IMG_SIZE = 64
    BATCH_SIZE = 32
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mentesi_mappa = f"eredmenyek/futtatas_{timestamp}"
    os.makedirs(mentesi_mappa, exist_ok=True) 
    print(f"--- Az eredmények ide fognak mentődni: {mentesi_mappa} ---")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root="./flower_photos", transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = FlowerVAE()
    logger = CSVLogger("logs", name="flower_vae") 
    
    trainer = pl.Trainer(
        max_epochs=20, 
        accelerator="cpu", 
        devices=1,
        logger=logger
    )
    print("--- Elindul a tanítás... ---")
    trainer.fit(model, train_loader)

    print(f"--- Eredmények mentése ide: {mentesi_mappa} ... ---")
    model.eval()
    
    try:
        metrics = pd.read_csv(f"{logger.log_dir}/metrics.csv")
        metrics = metrics.groupby('epoch').mean().reset_index()
        
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['epoch'], metrics['train_loss'], label='Hiba (Loss)', color='purple', linewidth=2)
        plt.title("A Modell Tanulási Görbéje (Loss)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Minél kisebb, annál jobb)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{mentesi_mappa}/01_tanitasi_gorbe.png")
        plt.close()
    except Exception as e:
        print(f"Nem sikerült a loss görbét megrajzolni: {e}")

    with torch.no_grad():
        imgs, _ = next(iter(train_loader))
        gen_imgs, _, _ = model(imgs)

        plt.figure(figsize=(12, 6))
        for i in range(4):
            plt.subplot(2, 4, i + 1)
            plt.title("Eredeti")
            plt.imshow((imgs[i].permute(1, 2, 0).numpy() + 1) / 2)
            plt.axis('off')
            
            plt.subplot(2, 4, i + 5)
            plt.title("Rekonstruált")
            plt.imshow((gen_imgs[i].permute(1, 2, 0).numpy() + 1) / 2)
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{mentesi_mappa}/02_rekonstrukcio.png")
        plt.close()

    with torch.no_grad():
        random_noise = torch.randn(4, 128) 
        new_flowers = model.decode(random_noise)
        
        plt.figure(figsize=(12, 4))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(f"Új Virág #{i+1}")
            plt.imshow((new_flowers[i].permute(1, 2, 0).numpy() + 1) / 2)
            plt.axis('off')
            
        plt.suptitle("Semmiből (Zajból) generált teljesen ÚJ virágok az AI-tól!", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{mentesi_mappa}/03_uj_viragok_generalva.png")
        plt.close()
        
    print(f"--- KÉSZ! Nézd meg az új mappádat: {mentesi_mappa} ---")

if __name__ == "__main__":
    main()