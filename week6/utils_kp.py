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
def getdatasets(data_dir, trn_csv, train_img_root, valid_pct=0.1, test_pct=0.1,verbose=False, random_seed=42 ):

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
