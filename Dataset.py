import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn

class CAERSRDataset(Dataset):
    def __init__(self, data_dir, resolution):
        self.data_dir = data_dir
        self.class_dict = self.create_class_dict() # {'class_name': 0, 'class_name': 1, ...}
        self.file_paths = self.get_file_paths() # ['data_dir/class_name/file_name', ...]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution),
        ])
        

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = Image.open(file_path).convert('RGB')
        label = self.get_label(file_path)

        image = self.transforms(image)
        label = torch.tensor(label) # not sure about bracets
        
        return image, label
    
    def get_classes(self):
        return len(self.class_dict)
    
    def create_class_dict(self):
        class_dict = {}
        folders = os.listdir(self.data_dir)
        for idx, folder in enumerate(folders):
            #ignore hidden folders
            if folder[0] != '.':
                class_dict[folder] = idx

        return class_dict
    
    def get_file_paths(self):
        file_paths = []
        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        file_paths.append(file_path)
        return file_paths
    
    def get_label(self, file_path):
        folder_name = os.path.basename(os.path.dirname(file_path))
        return self.class_dict[folder_name]