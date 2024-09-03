import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

load_model=False
save_model=True
checkpoint_gen="gen.pth"
checkpoint_disc="disc.pth"
device="cuda" if torch.cuda.is_available() else "cpu"
learning_rate=1e-4
num_epochs=100
batch_size=16
num_workers=4
high_res=96
low_res=high_res//4
img_channels=3

highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=low_res,height=low_res,interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0],std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=high_res,height=high_res),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)