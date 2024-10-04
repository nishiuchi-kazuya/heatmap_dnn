import glob
filelist = sorted(glob.glob('image*.png'))
train_size = int(len(filelist)*0.8)
train_ds_list = [(file, file.replace('image', 'label')) for file in filelist[:train_size]]
val_ds_list = [(file, file.replace('image', 'label')) for file in filelist[train_size:]]
for ds_type, ds_list in zip(['train', 'val'], [train_ds_list, val_ds_list]):    
    with open('{}.csv'.format(ds_type), 'w') as f:
        for d in ds_list:
            print(*d, sep=',', file=f)