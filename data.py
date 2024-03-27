import os 
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    """ X = Images and Y = masks"""

    train_x = sorted(glob(os.path.join(path, "training","images","*.tif")))
    train_y = sorted(glob(os.path.join(path, "training","1st_manual","*.gif")))

    test_x = sorted(glob(os.path.join(path, "test","images","*.tif")))
    test_y = sorted(glob(os.path.join(path, "test","1st_manual","*.gif")))

    return (train_x, train_y),(test_x, test_y)


'''
def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extraction des noms"""
        _, x_name = os.path.split(x)
        name, _ = os.path.splitext(x_name)
      
        """ Lecture des images et mask"""
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if augment == True:
            aug = HorizontalFlip(p=1.0)  
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]  

            aug = VerticalFlip(p=1.0)  
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]  
            
            aug =  ElasticTransform(p=1.0) 
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"] 
            
            aug = GridDistortion(p=1.0) 
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]  
            
            
            aug = OpticalDistortion(p=1.0)  
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]  
            
            aug = HorizontalFlip(p=1.0) 
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]  
            
            
            
            X = [x, x1, x2, x3,x4,x5]  
            Y = [y, y1, y2, y3,y4,y5]  # Correction ici
        else:
            X = [x]
            Y = [y]
            
        index = 0

        for i, m in zip(X, Y):  # Utilisation de x1 et m1
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if len(X) == 1:
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name = f"{name}.jpg"
            else:
                tmp_image_name = f"{name}_{index}.jpg"
                tmp_mask_name = f"{name}_{index}.jpg"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1
        # Vous devriez enlever le break ici pour traiter toutes les images
      
'''

if __name__ == "__main__":
    np.random.seed(42)

    data_path = "DRIVE"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train : {len(train_x)}) - {len(train_y)}")
    print(f"Test : {len(test_x)}) - {len(test_y)}")
    
    