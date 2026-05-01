DATASET_DIR = "data/raw"
PATCHES_DIR = "data/patches"

def make_patches(raw_dir: str = DATASET_DIR, patches_dir: str = PATCHES_DIR, patch_size: int = 64, patches_per_image: int = 20):
    """Crops random square patches from raw natural images."""
    pass

def main():
    make_patches()


if __name__ == "__main__":
    main()