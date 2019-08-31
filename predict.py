import cv2
import torch
from utils import resize_and_crop, normalize
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt

def plot_img_and_mask(raw_img, predict_mask, gnd_mask, real_coal, predict_coal):
    fig = plt.figure(figsize=(9, 3))
    # fig.tight_layout()
    # fig.set_size
    a = fig.add_subplot(1, 3, 1)
    a.set_title('Raw image')
    a.set_axis_off()
    plt.imshow(raw_img)

    b = fig.add_subplot(1, 3, 2)
    b.set_title('Groundtruth mask')
    b.set_xlabel('coal component: {:.3f}'.format(real_coal))
    plt.imshow(gnd_mask)

    c = fig.add_subplot(1, 3, 3)
    c.set_title('Predicted mask')
    c.set_xlabel('coal component: {:.3f}'.format(predict_coal))
    plt.imshow(predict_mask)
    plt.show()

if __name__ == '__main__':

    ori_w, ori_h = 852, 480
    dir_img = '/home/zhuzhu/Desktop/mid project/raw_data'
    dir_mask = '/home/zhuzhu/Desktop/mid project/groundtruth'
    id = '12'
    ori_img = cv2.imread(dir_img + '/%s.jpg'%id)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    mask = cv2.imread(dir_mask + '/' + id + '.png')

    red_mask = np.array(mask[:, :, 2] == 128)
    green_mask = np.array(mask[:, :, 1] == 128)
    true_mask = np.stack([red_mask.astype(np.float32), green_mask.astype(np.float32)])

    img[red_mask] = img[red_mask] * 0.8
    img[green_mask] = img[green_mask] * 1.2
    img[np.logical_and(mask[:, :, 2] != 128, mask[:, :, 1] != 128)] = 0.2 * \
                          img[np.logical_and(mask[:, :, 2] != 128, mask[:, :, 1] != 128)]

    img = resize_and_crop(img, scale=0.5)
    img = normalize(img)[None, None, :, :]
    img = torch.from_numpy(img).float()
    net = UNet(1, 2)
    net.eval()
    net.load_state_dict(torch.load('/media/zhuzhu/0C5809B80C5809B8/draft/unet/checkpoint/unet_0.854608765.pth', map_location='cpu'))
    predict = net(img).squeeze(0)

    mask_predict = (predict > 0.5).float().numpy()
    mask_blue = np.zeros(mask_predict.shape[1:])[np.newaxis, :]
    mask_predict = np.concatenate([mask_predict, mask_blue], axis=0)

    mask_predict = (mask_predict * 128).astype(np.uint8).transpose([1, 2, 0])
    mask_predict = cv2.resize(mask_predict, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

    target = np.zeros([ori_h, ori_w, 3]).astype(np.uint8)
    target[:, (ori_w-ori_h)//2:(ori_w-(ori_w-ori_h)//2), :] = mask_predict

    # target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

    # print('真实的煤占比：', np.sum(red_mask) / np.sum(true_mask))
    # print('预测出的煤占比：', mask_predict[:,:,0].sum()/mask_predict.sum())
    real_coal = np.sum(red_mask) / np.sum(true_mask)
    predict_coal = mask_predict[:,:,0].sum()/mask_predict.sum()
    plot_img_and_mask(ori_img, target, mask[:, :, ::-1], real_coal, predict_coal)
    # cv2.imwrite(id+'.png', target, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    # cv2.imshow('%s.png'%id, target)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
