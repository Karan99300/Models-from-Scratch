from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

class MapDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.list_files=os.listdir(self.root_dir)
        
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self,idx):
        img_file=self.list_files[idx]
        img_path=os.path.join(self.root_dir,img_file)
        img=np.array(Image.open(img_path))
        input_img=img[:,:600,:]
        target_img=img[:,600:,:]
        
        augmentations=both_transform(image=input_img,image0=target_img)
        input_img=augmentations["image"]
        target_img=augmentations["image0"]

        input_img=transform_only_input(image=input_img)["image"]
        target_img=transform_only_mask(image=target_img)["image"]
        
        return input_img,target_img
