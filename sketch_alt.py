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

RAW_IMG_DIR = "data/raw"
PATCHES_DIR = "data/patches"

RUN_NAME = "vae_alt"
OUTPUTS_DIR = f"outputs/{RUN_NAME}"
CHECKPOINT_PATH = f"{OUTPUTS_DIR}/vae_alt.pt"
SAMPLE_DIR = f"{OUTPUTS_DIR}/samples"

EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
LATENT_DIM = 32

BETA = 0.001
KL_WARMUP_EPOCHS = 5

PATCH_SIZE = 128
PATCHES_PER_IMAGE = 8
IMAGE_SIZE = 64

SAMPLE_TEMPERATURE = 0.7

def make_patches(raw_dir: str = RAW_IMG_DIR, patches_dir: str = PATCHES_DIR, patch_size: int = PATCH_SIZE, patches_per_image: int = PATCHES_PER_IMAGE):
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
    def __init__(self, image_dir="data/raw", image_size=IMAGE_SIZE):
        image_dir = Path(image_dir)
        self.image_paths = list(image_dir.glob("*.jpg"))
        if len(self.image_paths) == 0:
            raise ValueError(f"No .jpg images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        print(f"Loaded {len(self.image_paths)} images from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE):
        super().__init__()

        if image_size % 16 != 0:
            raise ValueError("IMAGE_SIZE must be divisible by 16, e.g. 64 or 128.")

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.feature_size = image_size // 16

        def enc_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def dec_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # Stronger encoder: 64 -> 32 -> 16 -> 8 -> 4
        self.encoder = nn.Sequential(
            enc_block(3, 64),
            enc_block(64, 128),
            enc_block(128, 256),
            enc_block(256, 512),
        )

        self.flatten_dim = 512 * self.feature_size * self.feature_size

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder: 4 -> 8 -> 16 -> 32 -> 64
        self.decoder = nn.Sequential(
            dec_block(512, 256),
            dec_block(256, 128),
            dec_block(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
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
        h = h.view(z.size(0), 512, self.feature_size, self.feature_size)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta: float = BETA):
    """Computes the VAE loss as a combination of reconstruction loss and KL divergence."""
    l1_loss = F.l1_loss(recon_x, x, reduction="mean")
    mse_loss = F.mse_loss(recon_x, x, reduction="mean")
    recon_loss = 0.85 * l1_loss + 0.15 * mse_loss

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
def generate_samples(checkpoint=CHECKPOINT_PATH, sample_dir=SAMPLE_DIR, latent_dim=LATENT_DIM, 
                     image_size=IMAGE_SIZE, num_samples=16, temperature=SAMPLE_TEMPERATURE):
    """Generates random samples from the trained VAE."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(latent_dim=latent_dim, image_size=image_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    z = temperature * torch.randn(num_samples, latent_dim).to(device)
    samples = model.decode(z)

    Path(sample_dir).mkdir(parents=True, exist_ok=True)

    save_path = Path(sample_dir) / f"generated_samples_temp_{temperature}.png"
    utils.save_image(samples, save_path, nrow=4)

    print(f"Saved generated samples to {save_path}")

def train(image_dir=RAW_IMG_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, latent_dim=LATENT_DIM, learning_rate=LEARNING_RATE, num_epochs=EPOCHS, beta=BETA, outputs_dir=OUTPUTS_DIR, ckpt_path=CHECKPOINT_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NatureDataset(image_dir=image_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    model = VAE(latent_dim=latent_dim).to(device)
    if Path(ckpt_path).exists():
        print(f"Loading existing model from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    Path(SAMPLE_DIR).mkdir(parents=True, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()

        current_beta = beta * min(1.0, (epoch + 1) / KL_WARMUP_EPOCHS)

        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} beta={current_beta:.5f}")
        for x in loop:
            x = x.to(device)

            recon_batch, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, x, mu, logvar, beta=current_beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix({
                "loss": loss.item(),
                "recon": recon_loss.item(),
                "kl": kl_loss.item(),
            })

        torch.save(model.state_dict(), ckpt_path)

        save_reconstructions(model, dataloader, device, save_path=Path(outputs_dir) / f"recon_epoch_{epoch + 1:03d}.png")

        model.eval()

        with torch.no_grad():
            z = SAMPLE_TEMPERATURE * torch.randn(16, latent_dim).to(device)
            samples = model.decode(z)

            utils.save_image(samples, Path(outputs_dir) / f"samples_epoch_{epoch + 1:03d}.png", nrow=4)

def main():
    # make_patches()

    train()

    generate_samples(
        checkpoint=CHECKPOINT_PATH,
        sample_dir=SAMPLE_DIR,
        latent_dim=LATENT_DIM,
        image_size=IMAGE_SIZE,
        num_samples=16,
        temperature=SAMPLE_TEMPERATURE,
    )

if __name__ == "__main__":
    main()