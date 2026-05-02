import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm

DATASET_DIR = "data/raw"
PATCHES_DIR = "data/patches"
OUTPUTS_DIR = "outputs"

EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
LATENT_DIM = 64
BETA = 0.001
PATCH_SIZE = 128
PATCHES_PER_IMAGE = 8

def make_patches(raw_dir: str = DATASET_DIR, patches_dir: str = PATCHES_DIR, patch_size: int = PATCH_SIZE, patches_per_image: int = PATCHES_PER_IMAGE):
    """Crops random square patches from raw natural images."""
    raw_dir, patches_dir = Path(raw_dir), Path(patches_dir)
    patches_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(list(raw_dir.glob("*.jpg"))):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        w, h  = img.size
        if w < patch_size or h < patch_size:
            continue

        for _ in range(patches_per_image):
            x = random.randint(0, w - patch_size)
            y = random.randint(0, h - patch_size)

            patch = img.crop((x, y, x + patch_size, y + patch_size))
            patch.save(patches_dir / f"{img_path.stem}_{x}_{y}.jpg")

class NatureDataset(Dataset):
    def __init__(self, image_dir="data/raw", image_size=64):
        image_dir = Path(image_dir)
        self.image_paths = list(image_dir.glob("*.jpg"))
        if len(self.image_paths) == 0:
            raise ValueError(f"No .jpg images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        print(f"Loaded {len(self.image_paths)} landscape images from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder: image -> compressed feature
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 64 -> 32
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),  # 32 -> 16
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1), # 16 -> 8
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, 2, 1), # 8 -> 4
            nn.ReLU(),
        )

        self.flatten_dim = 256 * 4 * 4

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder: latent vector -> image
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4 -> 8
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 -> 16
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16 -> 32
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32 -> 64
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(z.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta: float = BETA):
    """Computes the VAE loss as a combination of reconstruction loss and KL divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

@torch.no_grad()
def save_reconstructions(model, dataloader, device, save_path: str):
    model.eval()

    x = next(iter(dataloader))[:8].to(device)
    recon, _, _ = model(x)

    grid = torch.cat([x, recon], dim=0)
    utils.save_image(grid, save_path, nrow=8)

@torch.no_grad()
def generate_samples(checkpoint='outputs/vae_nature.pt', sample_dir="outputs/samples", latent_dim=LATENT_DIM, num_samples=16):
    """Generates new nature samples by sampling from the latent space of a trained VAE."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    z = torch.randn(num_samples, latent_dim).to(device)
    samples = model.decode(z)

    Path(sample_dir).mkdir(parents=True, exist_ok=True)

    save_path = Path(sample_dir) / "generated_samples.png"
    utils.save_image(samples, save_path, nrow=4)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NatureDataset(image_dir=DATASET_DIR, image_size=64)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    for epoch in range(EPOCHS):
        model.train()

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x in loop:
            x = x.to(device)

            recon_batch, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, x, mu, logvar, beta=BETA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix({
                "loss": loss.item(),
                "recon": recon_loss.item(),
                "kl": kl_loss.item()
            })
        
        ckpt_path = Path(OUTPUTS_DIR) / "vae_nature.pt"
        torch.save(model.state_dict(), ckpt_path)

        save_reconstructions(model, dataloader, device, save_path=Path(OUTPUTS_DIR) / f"recon_epoch_{epoch+1:03d}.png")

        model.eval()

        with torch.no_grad():
            z = torch.randn(16, LATENT_DIM).to(device)
            samples = model.decode(z)

            utils.save_image(
                samples,
                Path(OUTPUTS_DIR) / f"samples_epoch_{epoch + 1:03d}.png",
                nrow=4
            )

def main():
    make_patches()

    # train()

    # generate_samples(
    #     checkpoint="outputs/vae_nature.pt",
    #     sample_dir="outputs/samples",
    #     latent_dim=LATENT_DIM,
    #     num_samples=16
    # )

if __name__ == "__main__":
    main()