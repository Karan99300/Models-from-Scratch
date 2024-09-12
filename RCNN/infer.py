import cv2
import torch
from utils import load_checkpoint
from train import get_model
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import non_max_suppression
import numpy as np

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def inference(image,checkpoint):
    model=get_model().to(device)
    optimizer=optim.Adam(model.parameters())
    load_checkpoint(checkpoint,model,optimizer,learning_rate)
    
    ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast() 
    results=ss.process()
    
    copy1=image.copy()
    copy2=image.copy()
    
    positive_boxes=[]
    probs=[]
    
    for box in results:
        x1,y1,w,h=box
        x2,y2=x1+w,y1+h
        
        roi=image.copy()[y1:y2,x1:x2]
        roi=cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
        roi=preprocess_image(roi)
        
        with torch.no_grad():
            output=model(roi)
            prob=torch.softmax(output,dim=1)[0,1].item()
            class_pred=output.argmax(dim=1).item()
        
        if class_pred==1 and prob > 0.98:
            positive_boxes.append([x1, y1, x2, y2])
            probs.append(prob)
            cv2.rectangle(copy2, (x1, y1), (x2, y2), (255, 0, 0), 5)
    
        cleaned_boxes=non_max_suppression(positive_boxes,probs)
        
        for clean_box in cleaned_boxes:
            clean_x1, clean_y1, clean_x2, clean_y2 = map(int, clean_box)
            cv2.rectangle(copy1, (clean_x1, clean_y1), (clean_x2, clean_y2), (0, 255, 0), 3)
        
        plt.imshow(cv2.cvtColor(copy1, cv2.COLOR_BGR2RGB))
        plt.show()
        

def main(image_name,checkpoint):
    image=cv2.imread(image_name)
    return inference(image,checkpoint)

if __name__ == '__main__':
    checkpoint="rcnn_model.pth"
    device="cuda" if torch.cuda.is_available() else "cpu"
    save_model=True
    load_model=False
    learning_rate=1e-3
    image_name=""
    main(image_name,checkpoint)
