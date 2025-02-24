import numpy as np
from dataloader import CustomDataLoader
import os
import shutil
from sklearn.model_selection import train_test_split

#folder paths
dataset_dir = "C:/Users/Siegrain/Desktop/projects/EyeSpark/Dataset"
image_dir = f"{dataset_dir}/images"
json_dir = f"{dataset_dir}/annotations"
split_dir = f"{dataset_dir}/split_dataset"
corrupted_json_log = "corrupted_json.log"

#creating a folder for split dataset
for split in ["train", "val", "test"]:
    os.makedirs(f"{split_dir}/{split}/images", exist_ok=True)
    os.makedirs(f"{split_dir}/{split}/annotations", exist_ok=True)

#parameters
batch_size = 32
resize_dim = (64, 64)

#creating custom dataloader
loader = CustomDataLoader(image_dir, json_dir, batch_size=batch_size, shuffle=True, resize=resize_dim)

#data processing function
def preprocess_data(loader):
    preprocessed_images = []
    preprocessed_labels = []
    corrupted_files = []
    
    for images, labels in loader:
        if images is None or labels is None:
            continue  #skipping empty batches
        
        valid_labels = []
        valid_images = []
        
        for img, lbl in zip(images, labels):
            if isinstance(lbl, np.ndarray) and lbl.shape == (2,):  #now lbl is a numpy array already processed
                valid_labels.append(lbl)
                valid_images.append(img)
            else:
                print(f"[ERROR] annotation data not valid: {lbl}")
                corrupted_files.append(str(lbl))
                continue
        
        if valid_images:
            valid_images = np.array(valid_images)
            valid_labels = np.array(valid_labels)
            valid_images = (valid_images - 0.5) / 0.5  #normalization [-1,1]
            preprocessed_images.append(valid_images)
            preprocessed_labels.append(valid_labels)
    
    if len(preprocessed_images) == 0:
        print("[ERROR] no valid data has been preprocessed.")
        return np.array([]), np.array([])
    
    preprocessed_images = np.vstack(preprocessed_images)
    preprocessed_labels = np.vstack(preprocessed_labels)
    
    #saves corrupted json in a file
    if corrupted_files:
        with open(corrupted_json_log, "w", encoding="utf-8") as log_file:
            for file in corrupted_files:
                log_file.write(file + "\n")
        print(f"[INFO] corrupted json files saved in {corrupted_json_log}")
    
    return preprocessed_images, preprocessed_labels

#executing pre-processing
data_images, data_labels = preprocess_data(loader)

#dividing datased in Train (80%), Validation (10%), Test (10%)
X_train, X_temp, Y_train, Y_temp = train_test_split(data_images, data_labels, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

#moving data function
def move_files(indices, split):
    for idx in indices:
        img_name = f"{idx}.jpg"
        json_name = f"{idx}.json"
        img_src = os.path.join(image_dir, img_name)
        json_src = os.path.join(json_dir, json_name)
        img_dest = os.path.join(split_dir, split, "images", img_name)
        json_dest = os.path.join(split_dir, split, "annotations", json_name)
        
        if os.path.exists(img_src):
            shutil.copy(img_src, img_dest)
        if os.path.exists(json_src):
            shutil.copy(json_src, json_dest)

#moving files in the corrispondent folders
train_indices = range(len(X_train))
val_indices = range(len(X_train), len(X_train) + len(X_val))
test_indices = range(len(X_train) + len(X_val), len(X_train) + len(X_val) + len(X_test))

move_files(train_indices, "train")
move_files(val_indices, "val")
move_files(test_indices, "test")

#debug
print("datased divided and moved:")
print("Train set - Shape images:", X_train.shape, "Shape annotations:", Y_train.shape)
print("Validation set - Shape images:", X_val.shape, "Shape annotations:", Y_val.shape)
print("Test set - Shape images:", X_test.shape, "Shape annotations:", Y_test.shape)
