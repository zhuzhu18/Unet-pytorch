import os
import random
import cv2
import numpy as np
import math
import errno

def get_ids(dir):

    return (img_name[:-4] for img_name in os.listdir(dir))

def split_train_val(iddataset, val_percent=0.05):
    iddataset = list(iddataset)

    n_val = math.ceil(len(iddataset)*val_percent)
    random.shuffle(iddataset)

    return {'train': iddataset[:-n_val], 'val': iddataset[-n_val:]}

def resize_and_crop(img, scale):
    h, w = img.shape[:2]
    crop_w = (w - h) // 2
    img = img[:, crop_w:(w-crop_w)]
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    return img

def normalize(x):
    return x / 256

# ['coal', 'gangue']          # 红色是煤，绿色是矸石
def get_imgs_and_masks(ids, dir_img, dir_mask, scale=0.5, num_classes=2):
    for id in ids:
        img = cv2.imread(dir_img+'/'+id+'.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(dir_mask+'/'+id+'.png')

        red_mask = np.array(mask[:, :, 2] == 128)
        green_mask = np.array(mask[:, :, 1] == 128)
        true_mask = np.stack([red_mask.astype(np.float32), green_mask.astype(np.float32)])

        img[red_mask] = img[red_mask] * 0.8
        img[green_mask] = img[green_mask] * 1.2
        img[np.logical_and(mask[:, :, 2] != 128, mask[:, :, 1] != 128)] = 0.2 * \
                                  img[np.logical_and(mask[:, :, 2] != 128, mask[:, :, 1] != 128)]

        img = resize_and_crop(img, scale=scale)
        img = normalize(img)

        true_mask = true_mask.transpose([1, 2, 0])
        true_mask = resize_and_crop(true_mask, scale=scale)
        true_mask = true_mask.transpose([2, 0, 1])

        yield (img, true_mask)

def batch(iterable, batch_size):
    batch_data = []
    for i, j in enumerate(iterable):
        batch_data.append(j)
        if i % batch_size == batch_size - 1:
            yield batch_data
            batch_data = []
    if len(batch_data) != 0:
        yield batch_data

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
