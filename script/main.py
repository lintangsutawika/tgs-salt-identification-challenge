import os
import random
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import cv2

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable

from torchvision import models
from unet_models import SaltNet, UNet11

import lovasz_losses as L

from augment import *
from loss import *
from metric import *
from scheduler import *

path_train = '../input/train'

train_path_images = os.path.abspath(path_train + "/images/")
train_path_masks = os.path.abspath(path_train + "/masks/")

train_path_images_list = glob.glob(os.path.join(train_path_images, "*.png"))
train_path_masks_list = glob.glob(os.path.join(train_path_masks, "*.png"))

train_ids = next(os.walk(train_path_images))[2]
Y_target = np.zeros((len(train_ids), 1), dtype=int)
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    mask = imread(path_train + '/masks/' + id_).astype(np.bool_)
    Y_target[n] = np.sum(mask)/(101*101)*100
    # if np.sum(mask)/(101*101)*100 > 0:
    #     Y_target[n] = 1
    # else:
    #     Y_target[n] = 0

Y_target = [int(x) for x in pd.cut(Y_target.squeeze(), bins=[-0.1, 0.1, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], include_lowest=True, labels=['0','1','2','3','4','5','6','7','8','9','10'])]
SaltLevel = pd.DataFrame(data={'train_ids':train_ids, 'salt_class':Y_target})

class saltIDDataset(torch.utils.data.Dataset):

    def __init__(self, path_images, list_images, transforms=False, train="train", tta=True):
        self.train = train
        self.path_images = path_images
        self.list_images = list_images
        self.transforms = transforms
        self.tta = tta

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

                image, mask = do_resize2(image, mask, 101, 101)
                image, mask = do_center_pad_to_factor2(image, mask)
            else:
                image, mask = do_resize2(image, mask, 101, 101)
                image, mask = do_center_pad_to_factor2(image, mask)

            image = np.expand_dims(image, axis=2)
            mask = np.expand_dims(mask, axis=2)
            
            image = transformTensor(image).float()
            mask = transformTensor(mask).float()

            return (image, mask)

        else:
            if self.train == "valid":
                mask = cv2.imread(self.path_images + '/masks/' + self.list_images[idx],cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            else:
                mask = np.zeros([128,128])

            image, mask = do_resize2(image, mask, 101, 101)
            image, mask = do_center_pad_to_factor2(image, mask)

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

train_idx, valid_idx, SaltLevel_train, SaltLevel_valid = train_test_split(
    SaltLevel.index,
    SaltLevel,
    test_size=0.08, stratify=SaltLevel.salt_class)

fold_score = []
# sss = StratifiedShuffleSplit(n_splits=4, test_size=0.08)
# for cv_fold, (train_idx, valid_idx) in enumerate(sss.split(SaltLevel['train_ids'], SaltLevel['salt_class'])):
model = SaltNet()
# model = UNet11()
model.train()
if torch.cuda.is_available():
    model.cuda()


fold_score.append([0.0])
# sns.distplot(SaltLevel.salt_class.iloc[train_idx], label="Train")
# sns.distplot(SaltLevel.salt_class.iloc[valid_idx], label="Valid")
# plt.legend()
# plt.title("Salt Class Stratified Split Fold: {}".format(cv_fold))
# plt.savefig('salt_class_{}.png'.format(cv_fold), dpi=400)
# plt.close()

salt_ID_dataset_train = saltIDDataset(path_train, SaltLevel.train_ids.iloc[train_idx].values, transforms=True, train="train")
train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train, 
                                           batch_size=16, 
                                           shuffle=True,
                                           num_workers=1)

salt_ID_dataset_valid = saltIDDataset(path_train, SaltLevel.train_ids.iloc[valid_idx].values, transforms=False, train="valid")
val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_valid, 
                                           batch_size=8, 
                                           shuffle=True,
                                           num_workers=1)

epoch = 40
learning_rate = 1e-2

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
optimizer.zero_grad()
scheduler = CyclicScheduler(base_lr=0.001, max_lr=0.01, step=5., mode='triangular2', gamma=1., scale_fn=None, scale_mode='cycle') ##exp_range ##triangular2
best_iou = 0.0
for e in range(epoch):
    train_loss = []
    train_iou = []

    # for param in optimizer.param_groups:
    #     param['lr'] = scheduler.get_rate(e, epoch)
    # if e >= 100:
    #     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0001)

    model.train()
    with tqdm(train_loader) as pbar:
        for images, masks in pbar: 
            masks = masks.cuda()
            y_pred = model(Variable(images).cuda())

            prob = torch.sigmoid(y_pred).cpu().data.numpy()
            truth = masks.cpu().data.numpy()

            iou = do_kaggle_metric(prob, truth, threshold=0.5)
            train_iou.append(iou)

            # loss = torch.nn.BCEWithLogitsLoss()(y_pred, Variable(masks.cuda()))
            # loss = torch.nn.BCELoss()(y_pred, Variable(masks.cuda()))
            # loss = RobustFocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')
            loss = FocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')
            # loss = L.lovasz_hinge(y_pred, Variable(masks.cuda()), ignore=255)
            # loss = L.binary_xloss(y_pred, Variable(masks.cuda()), ignore=255)
            
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description("Loss: %.3f, IoU: %.3f, Progress" % (loss, iou))
        
    val_loss = []
    val_iou = []
    model.eval()
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

            prob = torch.sigmoid(y_pred).cpu().data.numpy()[:,:,13:-14,13:-14]
            truth = masks.cpu().data.numpy()[:,:,13:-14,13:-14]

            iou = do_kaggle_metric(prob, truth, threshold=0.5)
            val_iou.append(iou)

            # loss = torch.nn.BCEWithLogitsLoss()(y_pred, Variable(masks.cuda()))
            # loss = torch.nn.BCELoss()(y_pred, Variable(masks.cuda()))
            # loss = RobustFocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')
            loss = FocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')
            # loss = L.lovasz_hinge(y_pred, Variable(masks.cuda()), ignore=255)
            # loss = L.binary_xloss(y_pred, Variable(masks.cuda()), ignore=255)

            val_loss.append(loss.item())

            pbar.set_description("Loss: %.3f, IoU: %.3f, Progress" % (loss, iou))
    validation_iou = np.mean(val_iou)
    print("Epoch: %d, Train Loss: %.3f, Train IoU: %.3f,Val Loss: %.3f, Val IoU: %.3f" % (e, np.mean(train_loss), np.mean(train_iou), np.mean(val_loss), validation_iou))

    if validation_iou > best_iou:
        best_iou = validation_iou
        torch.save(model.state_dict(), "model_checkpoint_fold.pth")
        print("Better validation, model saved")
    else:
        pass

print("Training Finished, Best IoU: %.3f" % (best_iou))

#Choose best fold
model.load_state_dict(torch.load('model_checkpoint_fold.pth'))
#############################################################################################################
fold_score = []
model.train()
if torch.cuda.is_available():
    model.cuda()


epoch = 85
learning_rate = 0.0005
patience = 0
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
optimizer.zero_grad()
scheduler = CyclicScheduler(base_lr=0.001, max_lr=0.01, step=5., mode='triangular2', gamma=1., scale_fn=None, scale_mode='cycle') ##exp_range ##triangular2
best_iou = 0.0
for e in range(epoch):
    train_loss = []
    train_iou = []

    model.train()
    with tqdm(train_loader) as pbar:
        for images, masks in pbar: 
            masks = masks.cuda()
            y_pred = model(Variable(images).cuda())

            prob = torch.sigmoid(y_pred).cpu().data.numpy()
            truth = masks.cpu().data.numpy()

            iou = do_kaggle_metric(prob, truth, threshold=0.5)
            train_iou.append(iou)

            # loss = torch.nn.BCEWithLogitsLoss()(y_pred, Variable(masks.cuda()))
            # loss = torch.nn.BCELoss()(y_pred, Variable(masks.cuda()))
            # loss = RobustFocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')
            # loss = FocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')
            loss = L.lovasz_hinge(y_pred, Variable(masks.cuda()), ignore=255)
            # loss = L.binary_xloss(y_pred, Variable(masks.cuda()), ignore=255)
            
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description("Loss: %.3f, IoU: %.3f, Progress" % (loss, iou))
        
    val_loss = []
    val_iou = []
    model.eval()
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

            prob = torch.sigmoid(y_pred).cpu().data.numpy()[:,:,13:-14,13:-14]
            truth = masks.cpu().data.numpy()[:,:,13:-14,13:-14]

            iou = do_kaggle_metric(prob, truth, threshold=0.5)
            val_iou.append(iou)

            # loss = torch.nn.BCEWithLogitsLoss()(y_pred, Variable(masks.cuda()))
            # loss = torch.nn.BCELoss()(y_pred, Variable(masks.cuda()))
            # loss = RobustFocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')
            # loss = FocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')
            loss = L.lovasz_hinge(y_pred, Variable(masks.cuda()), ignore=255)
            # loss = L.binary_xloss(y_pred, Variable(masks.cuda()), ignore=255)

            val_loss.append(loss.item())

            pbar.set_description("Loss: %.3f, IoU: %.3f, Progress" % (loss, iou))
    validation_iou = np.mean(val_iou)
    print("Epoch: %d, Train Loss: %.3f, Train IoU: %.3f,Val Loss: %.3f, Val IoU: %.3f" % (e, np.mean(train_loss), np.mean(train_iou), np.mean(val_loss), validation_iou))

    if validation_iou > best_iou:
        best_iou = validation_iou
        torch.save(model.state_dict(), "model_checkpoint_finetune.pth")
        print("Better validation, model saved")
    else:
        patience += 1
        if patience == 6:
            print("learning_rate decreased")
            patience = 0
            learning_rate = learning_rate/2
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
            optimizer.zero_grad()

print("Training Finished, Best IoU: %.3f" % (best_iou))

#############################################################################################################
# model.load_state_dict(torch.load('model_checkpoint_finetune_fold{}.pth'.format(fold_score.index(max(fold_score)))))
# model.load_state_dict(torch.load('model_checkpoint_fold.pth'))
model.load_state_dict(torch.load('model_checkpoint_finetune.pth'))
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

        for i, img in enumerate(images):
            img_list.append(img)
            y_pred_one = y_pred[i] 
            y_pred_one = torch.sigmoid(y_pred_one).cpu().data.numpy()[:,13:-14,13:-14]
            y_pred_true_pairs.append((y_pred_one, masks[i].cpu().data.numpy()[:,13:-14,13:-14]))

thresholds = []
for threshold in np.linspace(0, 1, 51):
    ious = []
    for pair in y_pred_true_pairs:
        y_pred, mask = pair
        iou = do_kaggle_metric(y_pred, mask, threshold=threshold)
        ious.append(iou)
        
    thresholds.append([threshold, np.mean(ious)])
    print('Threshold: %.2f, Metric: %.3f' % (threshold, np.mean(ious)))

thresholds = np.asarray(thresholds)
best_threshold = thresholds[np.min(np.where(thresholds[:,1] == np.max(thresholds[:,1]))),0]
print('threshold: {}'.format(best_threshold))

# plt.figure(figsize=(20,20))
# for j in range(10):
#     q = j+1
#     plt.subplot(1,2*(1+10),q*2-1)
#     plt.imshow(y_pred_true_pairs[j+20][0][0] > best_threshold)
#     plt.subplot(1,2*(1+10),q*2)
#     plt.imshow(y_pred_true_pairs[j+20][1][0])
# plt.show()

path_test = '../input/test'
test_path_images = os.path.abspath(path_test + "/images/")
test_ids = next(os.walk(test_path_images))[2]
salt_ID_dataset_test = saltIDDataset(path_test, test_ids, transforms=False, train="test")
test_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_test, 
                                           batch_size=8, 
                                           shuffle=False,
                                           num_workers=1)

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

    for i, _ in enumerate(images):
        y_pred_one = y_pred[i] 
        y_pred_one = torch.sigmoid(y_pred_one)
        y_pred_one = y_pred_one.cpu().data.numpy()[0]
        y_pred_one = y_pred_one[13:-14,13:-14]
        y_pred_test.append(y_pred_one)

binary_prediction = (y_pred_test > best_threshold).astype(int)

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

all_masks = []
for p_mask in list(binary_prediction):
    p_mask = rle_encoding(p_mask)
    all_masks.append(' '.join(map(str, p_mask)))

test_path_images_list = glob.glob(os.path.join(test_path_images, "*.png"))
test_file_list = [f.split('/')[-1].split('.')[0] for f in tqdm(test_path_images_list)]
submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv('submission.csv', index = False)
