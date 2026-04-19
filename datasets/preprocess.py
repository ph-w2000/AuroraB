import numpy as np

import matplotlib.pyplot as plt
import os
import h5py 
from tqdm import tqdm
from PIL import Image
import xarray as xr
import pandas as pd



def calculate_dates(dates, start, end):
    if start < 0 or end >= len(dates):
        return False # Invalid range
    
    start_date = dates[start]
    end_date = dates[end]
    start_np = np.datetime64(start_date)
    end_np   = np.datetime64(end_date)

    delta = end_np - start_np                # numpy.timedelta64
    diff_in_minutes = delta / np.timedelta64(1, 'm')
    return diff_in_minutes

THRESHOLD = 3.0
data_dir = "../DiffCastB/data/MeteoNet/"

train_path = "../DiffCastB/data/MeteoNet/train/"
test_path = "../DiffCastB/data/MeteoNet/test/"
train_dirs = sorted(os.listdir(data_dir+"train/"))
test_dirs = sorted(os.listdir(data_dir+"test/"))

h5_data = "data/MeteoNet/MeteoNet.h5"
h5_file = h5py.File(h5_data, 'w')
h5_file.create_group('train')
h5_file.create_group('test')

seq_len = 0
total = 0
min_vals = 30000
min_key = ""

ds_2016 = None
ds_2017 = None
for dir in tqdm(train_dirs):
    img_paths = os.listdir(os.path.join(train_path, dir+"/"))
    long_seq = []
    long_dates = []

    for path in sorted(img_paths):
        img_path = os.path.join(train_path, dir, path)
        np_file = np.load(img_path, allow_pickle=True)
        data = np_file['data']
        dates = np_file['dates']
        long_seq.append(data)
        long_dates.append(dates)
        

    long_seq = np.concatenate(long_seq, axis=0)
    long_dates = np.concatenate(long_dates, axis=0)
    print("vil:",long_seq.shape, long_dates.shape)

    
    total += long_dates.shape[0]
    i = 0

    for i, idx in enumerate(range(len(long_seq))):
        if long_seq[idx].mean() > THRESHOLD and calculate_dates(long_dates, idx-2, idx+20) == 5*22:        #THRESHOLD'
            seq = long_seq[idx-2:idx+20]
            seq[seq == 255] = 0
            all_means = sum([frame.mean() for frame in seq])
            # if all_means > 3.5 * 30:
            key = str(seq_len)
            if all_means < min_vals:
                min_vals = all_means
                min_key = key
            h5_file['train'].create_dataset(str(key), data=seq, dtype='uint8', compression='lzf')

            dates = np.array([str(d).encode("utf-8") for d in long_dates[idx-2:idx+20]])
            h5_file['train'].create_dataset(str(key) + '_dates', data=dates, compression='lzf')
            seq_len += 1
            
h5_file['train'].create_dataset('all_len', data=seq_len)

print(total, seq_len)
print(min_vals)
print(min_key)

seq_len = 0
total = 0
min_vals = 30000
min_key = ""

ds_2018 = None
for dir in tqdm(test_dirs):
    img_paths = os.listdir(os.path.join(test_path,dir))
    long_dates = []
    long_seq = []

    for path in sorted(img_paths):
        img_path = os.path.join(test_path, dir, path)
        np_file = np.load(img_path, allow_pickle=True)
        data = np_file['data']
        dates = np_file['dates']
        long_seq.append(data)
        long_dates.append(dates)
    
    long_seq = np.concatenate(long_seq, axis=0)
    long_dates = np.concatenate(long_dates, axis=0)
    print("vil:",long_seq.shape, long_dates.shape)

    total += long_dates.shape[0]
    i = 0

    for i, idx in enumerate(range(len(long_seq))):
        if long_seq[idx].mean() > THRESHOLD and calculate_dates(long_dates, idx-2, idx+20) == 5*22:        #THRESHOLD'
            seq = long_seq[idx-2:idx+20]
            seq[seq == 255] = 0
            all_means = sum([frame.mean() for frame in seq])
            # if all_means > 3.5 * 30:
            key = str(seq_len)
            if all_means < min_vals:
                min_vals = all_means
                min_key = key
            h5_file['test'].create_dataset(str(key), data=seq, dtype='uint8', compression='lzf')

            dates = np.array([str(d).encode("utf-8") for d in long_dates[idx-2:idx+20]])
            h5_file['test'].create_dataset(str(key) + '_dates', data=dates, compression='lzf')
            seq_len += 1

h5_file['test'].create_dataset('all_len', data=seq_len)

h5_file.close()
print(total, seq_len)
print(min_vals)
print(min_key)


