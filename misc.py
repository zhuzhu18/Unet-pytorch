import cv2
import numpy as np
import matplotlib.pyplot as plt


# project = '/media/zhuzhu/ec114170-f406-444f-bee7-a3dc0a86cfa2/dataset/mid project'
#
# img = cv2.imread(project+'/raw_data/3.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# mask = cv2.imread(project+'/groundtruth/3.png')
#
#
# coal_mask = (mask[:, :, 2] == 128)
# gangue_mask = (mask[:, :, 1] == 128)
#
# img[coal_mask] = img[coal_mask] * 0.8
# img[gangue_mask] = img[gangue_mask] * 1.2
# img[np.logical_and(mask[:, :, 2] != 128, mask[:, :, 1] != 128)] = 0.2*\
#     img[np.logical_and(mask[:, :, 2] != 128, mask[:, :, 1] != 128)]
#
#
#
# hist_img = cv2.calcHist([img], [0], None, [256], ranges=(0, 256))
# hist_coal = cv2.calcHist([img], [0], coal_mask.astype(np.uint8), [256], (0, 256))
# hist_gangue = cv2.calcHist([img], [0], gangue_mask.astype(np.uint8), [256], (0, 256))
# plt.subplot(221)
# plt.imshow(img, cmap='gray')
# plt.subplot(222)
# plt.imshow(coal_mask, cmap='gray')
# plt.subplot(223)
# plt.imshow(gangue_mask, cmap='gray')
# plt.subplot(224)
#
# # plt.plot(hist_img, 'k-')
# plt.plot(hist_coal, 'r-')
# plt.plot(hist_gangue, 'g-')
# plt.xlim([0, 256])

# plt.show()

from sklearn.metrics import precision_recall_curve, roc_curve, classification_report
from utils import get_imgs_and_masks, get_ids, split_train_val
from unet import UNet
import torch

ori_w, ori_h = 852, 480
dir_img = '/home/zhuzhu/Desktop/mid project/raw_data'
dir_mask = '/home/zhuzhu/Desktop/mid project/groundtruth'
ids = get_ids(dir_img)
iddataset = split_train_val(ids, 0.05)

net = UNet(1, 2)
net.eval()
net.load_state_dict(torch.load('/media/zhuzhu/0C5809B80C5809B8/draft/unet/checkpoint/unet_0.854608765.pth', map_location='cpu'))
val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask)

c = 0
for i, b in enumerate(val):
    img = np.array(b[0]).astype(np.float32)
    mask = np.array(b[1]).astype(np.float32)

    with torch.no_grad():
        img = torch.from_numpy(img)[None, None, :, :]
        mask = torch.from_numpy(mask).unsqueeze(0)

        mask_pred = net(img)
        coal, gangue = mask_pred.data.numpy().reshape(2, -1)

        coal_fpr, coal_tpr, coal_th = roc_curve(mask.squeeze().numpy()[0].reshape(-1), coal)
        gangue_fpr, gangue_tpr, gangue_th_= roc_curve(mask.squeeze().numpy()[1].reshape(-1), gangue)
        c += 1
        plt.plot(coal_fpr ,coal_tpr, color='r', linestyle='-', label='coal')
        plt.plot(gangue_fpr, gangue_tpr, color='g', linestyle='-', label='gangue')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend(['coal', 'gangue'], loc='upper right')
        plt.show()
        if c == 1:
            exit(0)