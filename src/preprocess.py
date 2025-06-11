import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class DummyDataset(Dataset):
    def __init__(self, num_samples=10, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.captions = ["dummy caption {}".format(i) for i in range(num_samples)]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = Image.fromarray(np.uint8(np.random.rand(512,512,3)*255))
        if self.transform:
            image = self.transform(image)
        caption = self.captions[idx]
        return image, caption

def get_data_loader(num_samples=10, batch_size=2, test_mode=False):
    if test_mode:
        num_samples = 2
    transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
    dataset = DummyDataset(num_samples=num_samples, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
