# Import matplotlib for plotting learning rate vs loss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision import datasets, transforms
from pathlib import Path
import timm

import pandas as pd
from PIL import Image
def getdataframes(data_dir, trn_csv, train_img_root, valid_pct=0.1, test_pct=0.1,verbose=False, random_seed=42 ):

    train_csv_tmp = data_dir / 'train_split.csv'
    val_csv_tmp   = data_dir / 'valid_split.csv'
    test_csv_tmp  = data_dir / 'test_split.csv'

    #OK if not already created then create it
    if not Path(test_csv_tmp).exists():
        #get the complete train.csv
        df = pd.read_csv(trn_csv)

        indices = np.arange(len(df))
        np.random.shuffle(indices) #shuffle the indexes before splitting
        ten_percent = int(len(indices) * 0.1)

        #get total numbers of each set
        num_tst=int(test_pct*len(indices))
        num_val=int(valid_pct*len(indices))
        num_trn=len(indices) - (num_tst + num_val)

        #get indices
        tst_idx = indices[ :num_tst]
        val_idx = indices[num_tst:num_tst+num_val]
        trn_idx = indices[num_tst+num_val:]

        if verbose: print(f"Train: {len(trn_idx)}, Valid: {len(val_idx)}, Test: {len(tst_idx)}")

        df.iloc[tst_idx].to_csv(test_csv_tmp,  index=False)
        df.iloc[trn_idx].to_csv(train_csv_tmp, index=False)
        df.iloc[val_idx].to_csv(val_csv_tmp,   index=False)

    train_df=pd.read_csv(train_csv_tmp)
    valid_df=pd.read_csv(val_csv_tmp)
    test_df =pd.read_csv(test_csv_tmp)
    return train_df, valid_df, test_df


class PaddyMultitaskDataset(Dataset):
    """
    A custom PyTorch Dataset for multitask learning on the Paddy dataset.
    This dataset supports three tasks:
        1. Image classification of paddy disease labels.
        2. Image classification of paddy varieties.
        3. Regression of paddy plant age (normalized).
    Args:
        df (pd.DataFrame): DataFrame containing dataset metadata with columns:
            - 'image_id': Image file names.
            - 'label': Disease label for each image.
            - 'variety': Variety label for each image.
            - 'age': Age of the plant (numeric).
        img_root (Path or str): Root directory containing images, organized by label subfolders.
        transform (callable, optional): Optional transform to be applied on a sample image.
    Attributes:
        labels (List[str]): Sorted list of unique disease labels.
        varieties (List[str]): Sorted list of unique variety labels.
        label_to_idx (Dict[str, int]): Mapping from label names to integer indices.
        variety_to_idx (Dict[str, int]): Mapping from variety names to integer indices.
        age_mean (float): Mean of the 'age' column, used for normalization.
        age_std (float): Standard deviation of the 'age' column, used for normalization.
        num_label_classes (int): Number of unique disease labels.
        num_variety_classes (int): Number of unique variety labels.
    Returns:
        tuple: (image, variety_index, normalized_age, label_index)
            - image (Tensor): Transformed image tensor.
            - variety_index (Tensor): Integer index of the variety label.
            - normalized_age (Tensor): Normalized age value (float32).
            - label_index (Tensor): Integer index of the disease label.
    """

    def __init__(self, df, img_root: Path, transform=None):
        super().__init__()
        self.df = df
        self.img_root = Path(img_root)
        self.transform = transform

        self.labels = sorted(self.df['label'].astype(str).unique())
        self.varieties = sorted(self.df['variety'].astype(str).unique())

        #you have to convert strings to numbers for numerical processing
        self.label_to_idx = {s:i for i,s in enumerate(self.labels)}
        self.variety_to_idx = {s:i for i,s in enumerate(self.varieties)}
        self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')

        #lets get stats so we can normalize age if we want
        #should save stats from training set and apply those to valid and test
        self.age_mean = float(self.df['age'].mean())
        self.age_std = float(self.df['age'].std())

        self.num_label_classes = len(self.labels)
        self.num_variety_classes = len(self.varieties)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['image_id'])
        label_name = str(row['label'])
        variety_name = str(row['variety'])
        age_val = float(row['age'])

        #get and normalize image
        img_path = self.img_root / label_name / image_id
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        #get numerical label and variety
        y_label = self.label_to_idx[label_name]
        y_var = self.variety_to_idx[variety_name]

        # normalize age to mean 0 and std 1
        y_age = (age_val-self.age_mean) / self.age_std  

        return img, torch.tensor(y_var, dtype=torch.long), torch.tensor(y_age, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)

