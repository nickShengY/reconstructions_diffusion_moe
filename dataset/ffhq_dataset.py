from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

class FFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform

        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in sorted(os.listdir(root_dir))
            if fname.endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
