import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define image and label transforms
transform = transforms.Compose([
        transforms.ToTensor()
       ])

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target

# Make a PyTorch DataSet for use with the Dataloader class
class reduced_cifar(Dataset):
    def __init__(self, data, labels, transform=transform, target_transform=transform_target):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

        self.data = self.data.reshape((-1, 32, 32, 3)) # cifar has 32 x 32 images

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.data)