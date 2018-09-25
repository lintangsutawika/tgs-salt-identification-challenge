import os
import glob
import random
import pickle
import argparse

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

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--train', default="True")
parser.add_argument('--inference', default="True")
parser.add_argument('--ensemble', default="False")
parser.add_argument('--main', default="True")
parser.add_argument('--finetune', default=2)
parser.add_argument('--best_cv', default="True")
parser.add_argument('--batch_size', default=8)
args = parser.parse_args()

train_batch = int(args.batch_size)
right_pad = 27 #13
left_pad = 27 #14

path_train = '../input/train'

train_path_images = os.path.abspath(path_train + "/images/")
train_path_masks = os.path.abspath(path_train + "/masks/")

train_path_images_list = glob.glob(os.path.join(train_path_images, "*.png"))
train_path_masks_list = glob.glob(os.path.join(train_path_masks, "*.png"))

train_ids = np.array(next(os.walk(train_path_images))[2])
Y_target = np.zeros((len(train_ids), 1), dtype=float)
Y_alt = list(np.zeros((len(train_ids), 1), dtype=float))
pop_list = []
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    mask = imread(path_train + '/masks/' + id_).astype(np.bool_)
    pixel_sum = np.sum(mask)
    Y_target[n] = pixel_sum/(101.0*101.0)*100.0
    Y_alt[n] = pixel_sum
    if pixel_sum < 15 and pixel_sum != 0:
        pop_list.append(n)
    # if np.sum(mask)/(101*101)*100 > 0:
    #     Y_target[n] = 1
    # else:
    #     Y_target[n] = 0

# pop_list = []
train_ids = np.delete(train_ids, pop_list, None)
Y_target = np.delete(Y_target, pop_list, None)

Y_target = [int(x) for x in pd.cut(Y_target, bins=[-0.1, 0.1, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], include_lowest=True, labels=['0','1','2','3','4','5','6','7','8','9','10'])]
SaltLevel = pd.DataFrame(data={'train_ids':train_ids, 'salt_class':Y_target})

print("Dataset Size after removal: {}".format(len(train_ids)))

# train_idx, valid_idx, SaltLevel_train, SaltLevel_valid = train_test_split(
#     SaltLevel.index,
#     SaltLevel,
#     test_size=0.08, stratify=SaltLevel.salt_class)

#############################################################################################################
# Main Training
#############################################################################################################
if args.main == "True" and args.train == "True":
    print("Start Main Tuning")
    train_indexes = []
    valid_indexes = []
    fold_score = []
    sss = StratifiedShuffleSplit(n_splits=4, test_size=0.08)
    for cv_fold, (train_idx, valid_idx) in enumerate(sss.split(SaltLevel['train_ids'], SaltLevel['salt_class'])):
        print("Training for fold {}".format(cv_fold))
        model = SaltNet()
        # model = UNet11()
        model.train()
        if torch.cuda.is_available():
            model.cuda()

        train_indexes.append(train_idx)
        valid_indexes.append(valid_idx)

        # sns.distplot(SaltLevel.salt_class.iloc[train_idx], label="Train")
        # sns.distplot(SaltLevel.salt_class.iloc[valid_idx], label="Valid")
        # plt.legend()
        # plt.title("Salt Class Stratified Split Fold: {}".format(cv_fold))
        # plt.savefig('salt_class_{}.png'.format(cv_fold), dpi=400)
        # plt.close()

        salt_ID_dataset_train = saltIDDataset(path_train, SaltLevel.train_ids.iloc[train_idx].values, transforms=True, train="train")
        train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train, 
                                                   batch_size=train_batch, 
                                                   shuffle=True,
                                                   num_workers=1)

        salt_ID_dataset_valid = saltIDDataset(path_train, SaltLevel.train_ids.iloc[valid_idx].values, transforms=False, train="valid")
        val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_valid, 
                                                   batch_size=2, 
                                                   shuffle=True,
                                                   num_workers=1)

        epoch = 50
        learning_rate = 1e-2
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        scheduler = CyclicScheduler(base_lr=0.001, max_lr=0.01, step=5., mode='triangular2', gamma=1., scale_fn=None, scale_mode='cycle') ##exp_range ##triangular2
        optimizer.zero_grad()
        best_iou = 0.0
        for e in range(epoch):
            train_loss = []
            train_iou = []

            learning_rate = scheduler.get_rate(e, epoch)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

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
                    # loss = L.lovasz_hinge(y_pred.squeeze(), masks.squeeze().cuda(), per_image=True, ignore=None)
                    loss = FocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')
                    
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

                    prob = torch.sigmoid(y_pred)[:,:,right_pad:-left_pad,right_pad:-left_pad]
                    truth = masks[:,:,right_pad:-left_pad,right_pad:-left_pad]
                    
                    prob = F.interpolate(prob, size=(101,101)).cpu().data.numpy()
                    truth = F.interpolate(truth, size=(101,101)).cpu().data.numpy()

                    iou = do_kaggle_metric(prob, truth, threshold=0.5)
                    val_iou.append(iou)

                    # loss = torch.nn.BCEWithLogitsLoss()(y_pred, Variable(masks.cuda()))
                    # loss = torch.nn.BCELoss()(y_pred, Variable(masks.cuda()))
                    # loss = RobustFocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')
                    # loss = L.lovasz_hinge(y_pred.squeeze(), masks.squeeze().cuda(), per_image=True, ignore=None)
                    loss = FocalLoss2d()(y_pred, Variable(masks.cuda()), type='sigmoid')

                    val_loss.append(loss.item())

                    pbar.set_description("Loss: %.3f, IoU: %.3f, Progress" % (loss, iou))
            validation_iou = np.mean(val_iou)
            print("Epoch: %d, Train Loss: %.3f, Train IoU: %.3f,Val Loss: %.3f, Val IoU: %.3f" % (e, np.mean(train_loss), np.mean(train_iou), np.mean(val_loss), validation_iou))

            if validation_iou > best_iou:
                best_iou = validation_iou
                torch.save(model.state_dict(), "model_checkpoint_fold_{}.pth".format(cv_fold))
                print("Better validation, model saved")
            else:
                pass
        fold_score.append(best_iou)

    best_fold = fold_score.index(max(fold_score))
    print("Training Finished, Best IoU: %.3f at fold {}".format(best_fold) % (max(fold_score)))
    with open('indexes.pkl', 'wb') as f:
        pickle.dump([max(fold_score), train_indexes, valid_indexes], f)

#############################################################################################################
#Fine Tuning #1
#############################################################################################################
if int(args.finetune) >= 1 and args.train == "True":
    print("Start Fine Tuning #1")
    with open('indexes.pkl', 'rb') as f:
        saved_best_cv, train_indexes, valid_indexes= pickle.load(f)

    if args.best_cv == "True":
        selected_cv = saved_best_cv
    else:
        selected_cv = int(args.best_cv)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load('model_checkpoint_fold_{}.pth'.format(selected_cv)))
    else:
        model.load_state_dict(torch.load('model_checkpoint_fold_{}.pth'.format(selected_cv), map_location='cpu'))

    train_idx = train_indexes[selected_cv]
    valid_idx = valid_indexes[selected_cv]

    salt_ID_dataset_train = saltIDDataset(path_train, SaltLevel.train_ids.iloc[train_idx].values, transforms=True, train="train")
    train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train, 
                                               batch_size=train_batch, 
                                               shuffle=True,
                                               num_workers=1)

    salt_ID_dataset_valid = saltIDDataset(path_train, SaltLevel.train_ids.iloc[valid_idx].values, transforms=False, train="valid")
    val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_valid, 
                                               batch_size=2, 
                                               shuffle=True,
                                               num_workers=1)

    epoch = 85
    learning_rate = 0.005
    patience = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    optimizer.zero_grad()
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

                loss = L.lovasz_hinge(y_pred.squeeze(), masks.squeeze().cuda(), per_image=True, ignore=None)
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

                prob = torch.sigmoid(y_pred)[:,:,right_pad:-left_pad,right_pad:-left_pad]
                truth = masks[:,:,right_pad:-left_pad,right_pad:-left_pad]
                
                prob = F.interpolate(prob, size=(101,101)).cpu().data.numpy()
                truth = F.interpolate(truth, size=(101,101)).cpu().data.numpy()

                iou = do_kaggle_metric(prob, truth, threshold=0.5)
                val_iou.append(iou)

                loss = L.lovasz_hinge(y_pred.squeeze(), masks.squeeze().cuda(), per_image=True, ignore=None)
                val_loss.append(loss.item())

                pbar.set_description("Loss: %.3f, IoU: %.3f, Progress" % (loss, iou))
        validation_iou = np.mean(val_iou)
        print("Epoch: %d, Train Loss: %.3f, Train IoU: %.3f,Val Loss: %.3f, Val IoU: %.3f" % (e, np.mean(train_loss), np.mean(train_iou), np.mean(val_loss), validation_iou))

        if validation_iou > best_iou:
            best_iou = validation_iou
            torch.save(model.state_dict(), "model_checkpoint_finetune_1_fold_{}.pth".format(best_cv))
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
# Fine Tuning #2
#############################################################################################################
if int(args.finetune) >= 2 and args.train == "True":
    print("Start Fine Tuning #2")

    with open('indexes.pkl', 'rb') as f:
        saved_best_cv, train_indexes, valid_indexes= pickle.load(f)

    if args.best_cv == "True":
        selected_cv = saved_best_cv
    else:
        selected_cv = int(args.best_cv)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load("model_checkpoint_finetune_1_fold_{}.pth".format(selected_cv)))
    else:
        model.load_state_dict(torch.load("model_checkpoint_finetune_1_fold_{}.pth".format(selected_cv)), map_location='cpu')

    train_idx = train_indexes[selected_cv]
    valid_idx = valid_indexes[selected_cv]

    salt_ID_dataset_train = saltIDDataset(path_train, SaltLevel.train_ids.iloc[train_idx].values, transforms=True, train="train")
    train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train, 
                                               batch_size=train_batch, 
                                               shuffle=True,
                                               num_workers=1)

    salt_ID_dataset_valid = saltIDDataset(path_train, SaltLevel.train_ids.iloc[valid_idx].values, transforms=False, train="valid")
    val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_valid, 
                                               batch_size=2, 
                                               shuffle=True,
                                               num_workers=1)

    epoch = 60
    learning_rate = 0.001
    patience = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    optimizer.zero_grad()
    best_iou = 0.0
    for e in range(epoch):
        train_loss = []
        train_iou = []

        model.train()
        model.apply(FreezeBatchNorm)
        with tqdm(train_loader) as pbar:
            for images, masks in pbar: 
                masks = masks.cuda()
                y_pred = model(Variable(images).cuda())

                prob = torch.sigmoid(y_pred).cpu().data.numpy()
                truth = masks.cpu().data.numpy()

                iou = do_kaggle_metric(prob, truth, threshold=0.5)
                train_iou.append(iou)

                loss = L.lovasz_hinge(y_pred.squeeze(), masks.squeeze().cuda(), per_image=True, ignore=None)
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

                prob = torch.sigmoid(y_pred)[:,:,right_pad:-left_pad,right_pad:-left_pad]
                truth = masks[:,:,right_pad:-left_pad,right_pad:-left_pad]
                
                prob = F.interpolate(prob, size=(101,101)).cpu().data.numpy()
                truth = F.interpolate(truth, size=(101,101)).cpu().data.numpy()

                iou = do_kaggle_metric(prob, truth, threshold=0.5)
                val_iou.append(iou)

                loss = L.lovasz_hinge(y_pred.squeeze(), masks.squeeze().cuda(), per_image=True, ignore=None)
                val_loss.append(loss.item())

                pbar.set_description("Loss: %.3f, IoU: %.3f, Progress" % (loss, iou))
        validation_iou = np.mean(val_iou)
        print("Epoch: %d, Train Loss: %.3f, Train IoU: %.3f,Val Loss: %.3f, Val IoU: %.3f" % (e, np.mean(train_loss), np.mean(train_iou), np.mean(val_loss), validation_iou))

        if validation_iou > best_iou:
            best_iou = validation_iou
            torch.save(model.state_dict(), "model_checkpoint_finetune.pth_2_fold_{}.pth".format(cv_fold))
            print("Better validation, model saved")
        else:
            patience += 1
            if patience == 6:
                print("learning_rate decreased")
                patience = 0
                learning_rate = learning_rate/2
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
                optimizer.zero_grad()
    fold_score.append(best_iou)

#############################################################################################################
# Inference
#############################################################################################################

if args.inference == "True":

    path_test = '../input/test'
    test_path_images = os.path.abspath(path_test + "/images/")
    test_ids = next(os.walk(test_path_images))[2]
    salt_ID_dataset_test = saltIDDataset(path_test, test_ids, transforms=False, train="test")
    test_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_test, 
                                               batch_size=8, 
                                               shuffle=False,
                                               num_workers=1)

    with open('indexes.pkl', 'rb') as f:
        saved_best_cv, train_indexes, valid_indexes= pickle.load(f)
    
    if args.ensemble == "True":
        for n in range(4):
            valid_idx = valid_indexes[n]
            salt_ID_dataset_valid = saltIDDataset(path_train, SaltLevel.train_ids.iloc[valid_idx].values, transforms=False, train="valid")
            val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_valid, 
                                                       batch_size=2, 
                                                       shuffle=True,
                                                       num_workers=1)
            if args.finetune == 2:
                model_name = "model_checkpoint_finetune_2_fold_{}.pth".format(n)
            elif args.finetune == 1:
                model_name = "model_checkpoint_finetune_1_fold_{}.pth".format(n)
            elif args.main == "True":
                model_name = "model_checkpoint_fold_{}.pth".format(n)

            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_name))
            else:
                model.load_state_dict(torch.load(model_name), map_location='cpu')

            if n == saved_best_cv:
                best_threshold = find_best_threshold(model, val_loader)

            if n == 0:
                y_pred_test = infer_prediction(model, test_loader)
            else:
                y_pred_test += infer_prediction(model, test_loader)

        y_pred_test = y_pred_test/4
        binary_prediction = (y_pred_test > best_threshold).astype(int)

    else:
        valid_idx = valid_indexes[n]
        salt_ID_dataset_valid = saltIDDataset(path_train, SaltLevel.train_ids.iloc[valid_idx].values, transforms=False, train="valid")
        val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_valid, 
                                                   batch_size=2, 
                                                   shuffle=True,
                                                   num_workers=1)
        if args.finetune == 2:
            model_name = "model_checkpoint_finetune_2_fold_{}.pth".format(saved_best_cv)
        elif args.finetune == 1:
            model_name = "model_checkpoint_finetune_1_fold_{}.pth".format(saved_best_cv)
        elif args.main == "True":
            model_name = "model_checkpoint_fold_{}.pth".format(saved_best_cv)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_name))
        else:
            model.load_state_dict(torch.load(model_name), map_location='cpu')

        best_threshold = find_best_threshold(model, val_loader)

        y_pred_test = infer_prediction(model, test_loader)
        binary_prediction = (y_pred_test > best_threshold).astype(int)

    all_masks = []
    for p_mask in list(binary_prediction):
        p_mask = rle_encoding(p_mask)
        all_masks.append(' '.join(map(str, p_mask)))

    test_path_images_list = glob.glob(os.path.join(test_path_images, "*.png"))
    test_file_list = [f.split('/')[-1].split('.')[0] for f in tqdm(test_path_images_list)]
    submit = pd.DataFrame([test_file_list, all_masks]).T
    submit.columns = ['id', 'rle_mask']
    submit.to_csv('submission.csv', index = False)

else:
    print("No Inference, program finished")
    sys.exit(1)