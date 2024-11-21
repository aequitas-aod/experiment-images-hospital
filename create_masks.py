import cv2
from dataset import CropsDataset
import numpy as np
import os


def get_segmented(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    
    #img = h if h.max() - np.quantile(np.unique(h), .8) < 50 else 255 - s
    img = 255-s
    
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #if len(np.where(thresh == 0)[0]) >= len(np.where(thresh == 255)[0]):
        #thresh = 255 - thresh

    return thresh

def create_masks(disease):
    #disease = 'esantema-polimorfo-like'
    folder_name = os.path.join(disease, 'crops_masks')
    os.makedirs(folder_name, exist_ok = True)
    dataset = CropsDataset(disease)
    for i in dataset.data.keys():
        cv2.imwrite(f'{folder_name}/{os.path.basename(i)}_mask.png', get_segmented(dataset[i]))