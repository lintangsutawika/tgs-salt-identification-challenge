import cv2
import torch

from tqdm import tqdm

import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

import numpy as np

from metric import *
from augment import *

right_pad = 27 #13
left_pad = 27 #14

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def FreezeBatchNorm(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()

class saltIDDataset(torch.utils.data.Dataset):

    def __init__(self, path_images, list_images, transforms=False, train="train", tta=True, depth=False):
        self.image_size = 256 #128
        self.resize_to = 202 #101
        self.factor = 64 #32
        self.train = train
        self.path_images = path_images
        self.list_images = list_images
        self.transforms = transforms
        self.tta = tta
        self.depth = depth

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        image = cv2.imread(self.path_images + '/images/' + self.list_images[idx],cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
        transformTensor = transforms.ToTensor()
        if self.train == "train":
            mask = cv2.imread(self.path_images + '/masks/' + self.list_images[idx],cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            if self.transforms is True:
                if np.random.rand() < 0.5:
                    image, mask = do_horizontal_flip2(image, mask)

                if np.random.rand() < 0.5:
                    choice = np.random.choice(4)
                    if choice == 0:
                        image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)
                    elif choice == 1:
                        image, mask = do_horizontal_shear2(image, mask, dx=np.random.uniform(-0.07, 0.07))
                    elif choice == 2:
                        image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(-10,10))
                    elif choice == 3:
                        image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0,0.15))

                if np.random.rand() < 0.5:
                    choice = np.random.choice(3)
                    if choice == 0:
                        image = do_brightness_shift(image, np.random.uniform(-0.1, +0.1))
                    elif choice == 1:
                        image = do_brightness_multiply(image, np.random.uniform(1-0.08,1+0.08))
                    elif choice == 2:
                        image = do_gamma(image, np.random.uniform(1-0.08,1+0.08))

                image, mask = do_resize2(image, mask, self.resize_to, self.resize_to)
                image, mask = do_center_pad_to_factor2(image, mask, factor=self.factor)
                if self.depth == True:
                    image = add_depth_channels(image)
            else:
                image, mask = do_resize2(image, mask, self.resize_to, self.resize_to)
                image, mask = do_center_pad_to_factor2(image, mask, factor=self.factor)
                if self.depth == True:
                    image = add_depth_channels(image)

            image = np.expand_dims(image, axis=2)
            mask = np.expand_dims(mask, axis=2)
            
            image = transformTensor(image).float()
            mask = transformTensor(mask).float()

            return (image, mask)

        else:
            if self.train == "valid":
                mask = cv2.imread(self.path_images + '/masks/' + self.list_images[idx],cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            else:
                mask = np.zeros([self.image_size,self.image_size])

            image, mask = do_resize2(image, mask, self.resize_to, self.resize_to)
            image, mask = do_center_pad_to_factor2(image, mask, factor=64)
            if self.depth == True:
                image = add_depth_channels(image)

            if self.tta == True:            
                image_flip, mask_flip = do_horizontal_flip2(image, mask)
                image_flip = np.expand_dims(image_flip, axis=2)
                image_flip = transformTensor(image_flip).float()

                mask_flip = np.expand_dims(mask_flip, axis=2)
                mask_flip = transformTensor(mask_flip).float()
            
            image = np.expand_dims(image, axis=2)
            mask = np.expand_dims(mask, axis=2)

            image = transformTensor(image).float()
            mask = transformTensor(mask).float()

            if self.tta == True:
                return ((image, image_flip), (mask, mask_flip))
            else:
                return (image, mask)

def find_best_threshold(model, val_loader):
    model.eval()
    y_pred_true_pairs = []
    img_list = []
    with tqdm(val_loader) as pbar:
        for images, masks in pbar:
            if len(images) == 2:
                image_ori, image_rev = images
                mask_ori, mask_rev = masks
                if torch.cuda.is_available():
                    image_ori, image_rev, mask_ori, mask_rev = image_ori.cuda(), image_rev.cuda(), mask_ori.cuda(), mask_rev.cuda()

                y_pred_rev = model(Variable(image_rev)).flip(3)
                y_pred_ori = model(Variable(image_ori))
                y_pred = (y_pred_ori+ y_pred_rev)/2
                masks = mask_ori
            else:
                if torch.cuda.is_available():
                    images, masks = images.cuda(), masks.cuda()
                y_pred = model(Variable(images))

            y_pred = torch.sigmoid(y_pred)[:,:,right_pad:-left_pad,right_pad:-left_pad]
            y_pred = F.interpolate(y_pred, size=(101,101)).cpu().data.numpy()

            masks = masks[:,:,right_pad:-left_pad,right_pad:-left_pad]
            masks = F.interpolate(masks, size=(101,101)).cpu().data.numpy()
            for i, img in enumerate(images):
                img_list.append(img)
                y_pred_one = y_pred[i] 
                mask = masks[i]
                y_pred_true_pairs.append((y_pred_one, mask))

    thresholds = []
    for threshold in np.linspace(0, 1, 51):
        ious = []
        for pair in y_pred_true_pairs:
            y_pred, mask = pair
            iou = do_kaggle_metric(y_pred, mask, threshold=threshold)
            ious.append(iou)
            
        thresholds.append([threshold, np.mean(ious)])
        # print('Threshold: %.2f, Metric: %.3f' % (threshold, np.mean(ious)))

    thresholds = np.asarray(thresholds)
    best_threshold = thresholds[np.min(np.where(thresholds[:,1] == np.max(thresholds[:,1]))),0]
    print('threshold: {}, Score: %.3f'.format(best_threshold) %(np.max(thresholds[:,1])))
    return best_threshold

def infer_prediction(model, test_loader):
    model.eval()
    y_pred_test = []
    for images, masks in tqdm(test_loader):
        if len(images) == 2:
            image_ori, image_rev = images
            mask_ori, mask_rev = masks
            if torch.cuda.is_available():
                image_ori, image_rev, mask_ori, mask_rev = image_ori.cuda(), image_rev.cuda(), mask_ori.cuda(), mask_rev.cuda()

            y_pred_rev = model(Variable(image_rev)).flip(3)
            y_pred_ori = model(Variable(image_ori))
            y_pred = (y_pred_ori+ y_pred_rev)/2
            masks = mask_ori
            images = image_ori
        else:
            if torch.cuda.is_available():
                images, masks = images.cuda(), masks.cuda()
            y_pred = model(Variable(images))

        y_pred = torch.sigmoid(y_pred)[:,:,right_pad:-left_pad,right_pad:-left_pad]
        y_pred = F.interpolate(y_pred, size=(101,101)).cpu().data.numpy()
        for i, _ in enumerate(images):
            y_pred_one = y_pred[i] 
            y_pred_test.append(y_pred_one)

    return np.array(y_pred_test)

# plt.figure(figsize=(20,20))
# for j in range(10):
#     q = j+1
#     plt.subplot(1,2*(1+10),q*2-1)
#     plt.imshow(y_pred_true_pairs[j+20][0][0] > best_threshold)
#     plt.subplot(1,2*(1+10),q*2)
#     plt.imshow(y_pred_true_pairs[j+20][1][0])
# plt.show()

