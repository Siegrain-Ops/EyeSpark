import os
import json
import numpy as np
import random
from PIL import Image, UnidentifiedImageError

class CustomDataLoader:
    def __init__(self, image_dir, json_dir, batch_size=32, shuffle=True, resize=(64, 64)):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize = resize
        
        #loadin the files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
        
        assert len(self.image_files) == len(self.json_files), "Images and json files are not the same amount"
        
        self.data_pairs = list(zip(self.image_files, self.json_files))
        if self.shuffle:
            random.shuffle(self.data_pairs)
        
        self.index = 0
        self.num_batches = len(self.data_pairs) // self.batch_size
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        self.index = 0
        if self.shuffle:
            random.shuffle(self.data_pairs)
        return self
    
    def __next__(self):
        if self.index >= len(self.data_pairs):
            raise StopIteration
        
        batch_images = []
        batch_labels = []
        
        for i in range(self.batch_size):
            if self.index >= len(self.data_pairs):
                break
            img_name, json_name = self.data_pairs[self.index]
            img_path = os.path.join(self.image_dir, img_name)
            json_path = os.path.join(self.json_dir, json_name)
            
            try:
                #loadin image
                image = Image.open(img_path).convert('RGB')
                image = image.resize(self.resize)
                image = np.array(image) / 255.0  #normalizing
            except (UnidentifiedImageError, IOError):
                print(f"image not valid: {img_path}")
                self.index += 1
                continue
            
            try:
                #loading json and acces to data
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "eye_details" in data and "look_vec" in data["eye_details"]:
                    look_vec = data["eye_details"]["look_vec"]
                    
                    #check if look_vec is a string, conversion to a float list
                    if isinstance(look_vec, str):
                        look_vec = look_vec.strip("() ").split(",")
                        look_vec = [float(val) for val in look_vec]
                    
                    if isinstance(look_vec, list) and len(look_vec) >= 2:
                        gaze_x, gaze_y = look_vec[0], look_vec[1]
                        label = np.array([gaze_x, gaze_y], dtype=np.float32)
                        
                        #convert value in list before printing
                        print("JSON loaded:", json.dumps({"look_vec": label.tolist()}, indent=2))
                    else:
                        print(f"[ERROR] wrong structre of 'look_vec' in JSON: {json.dumps(data, indent=2)}")
                        self.index += 1
                        continue
                else:
                    print(f"[ERROR] key 'eye_details' or 'look_vec' missing in JSON: {json.dumps(data, indent=2)}")
                    self.index += 1
                    continue
            except Exception as e:
                print(f"[ERROR] reading JSON failed: {json_path}. {e}")
                self.index += 1
                continue
            
            batch_images.append(image)
            batch_labels.append(label)
            self.index += 1
        
        if len(batch_images) == 0:
            raise StopIteration
        
        return np.array(batch_images), np.array(batch_labels)

# Esempio di utilizzo
if __name__ == "__main__":
    image_dir = "C:/Users/Siegrain/Desktop/projects/EyeSpark/Dataset/images"
    json_dir = "C:/Users/Siegrain/Desktop/projects/EyeSpark/Dataset/annotations"
    
    loader = CustomDataLoader(image_dir, json_dir, batch_size=32, shuffle=True)
    
    for images, labels in loader:
        print("Images Batch:", images.shape)
        print("Annotations Batch:", labels.shape)
        break
