import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import iou

class RCNNDataset(Dataset):
    def __init__(self, image_path, annotations_path, transform=None):
        self.image_path = image_path
        self.annotations_path = annotations_path
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.images, self.labels = self._load_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_dataset(self):
        images = []
        labels = []

        for annotation_file in os.listdir(self.annotations_path):
            if not annotation_file.startswith("airplane"):
                continue

            try:
                filename = annotation_file.split(".")[0] + ".jpg"
                image_path = os.path.join(self.image_path, filename)
                image = cv2.imread(image_path)
                
                df = pd.read_csv(os.path.join(self.annotations_path, annotation_file))
                gt_values = []

                for _, row in df.iterrows():
                    coords = list(map(int, row[0].split()))
                    gt_values.append({"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3]})

                self.ss.setBaseImage(image)
                self.ss.switchToSelectiveSearchFast()
                ss_results = self.ss.process()

                pos_counter = 0
                neg_counter = 0

                for idx, (x, y, w, h) in enumerate(ss_results):
                    if idx >= 2000 or (pos_counter >= 30 and neg_counter >= 30):
                        break

                    for gt_val in gt_values:
                        IOU = iou(gt_val, {"x1": x, "x2": x+w, "y1": y, "y2": y+h})
                        
                        if IOU >0.5 and pos_counter < 30:
                            t_image = image[y:y+h, x:x+w]
                            t_image = cv2.resize(t_image, (224, 224), interpolation=cv2.INTER_AREA)
                            images.append(t_image)
                            labels.append(1)
                            pos_counter += 1
                        elif IOU < 0.3 and neg_counter < 30:
                            t_image = image[y:y+h, x:x+w]
                            t_image = cv2.resize(t_image, (224, 224), interpolation=cv2.INTER_AREA)
                            images.append(t_image)
                            labels.append(0)
                            neg_counter += 1

            except Exception as e:
                print(f"Error in {filename}: {e}")
                continue

        return images, labels

def get_rcnn_dataset(image_path, annotations_path):
    cv2.setUseOptimized(True)
    return RCNNDataset(image_path, annotations_path)

if __name__ == '__main__':
    image_path = "/content/drive/My Drive/AI content/RCNN-master/Images"
    annotations_path = "/content/drive/My Drive/AI content/RCNN-master/Airplanes_Annotations"
    dataset = get_rcnn_dataset(image_path, annotations_path)
    print(f"Dataset size: {len(dataset)}")
    print(f"First item: {dataset[0]}")