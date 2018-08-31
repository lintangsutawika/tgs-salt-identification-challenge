import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from glob import glob

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable

from torchvision import models
from unet_models import AlbuNet, UNet11, UNetVGG16, UNetResNet, Unet

path_train = '../input/train'

train_path_images = os.path.abspath(path_train + "/images/")
train_path_masks = os.path.abspath(path_train + "/masks/")

train_path_images_list = glob(os.path.join(train_path_images, "*.png"))
train_path_masks_list = glob(os.path.join(train_path_masks, "*.png"))

train_ids = next(os.walk(train_path_images))[2]
Y_target = np.zeros((len(train_ids), 1), dtype=int)
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    mask = imread(path_train + '/masks/' + id_).astype(np.bool_)
    Y_target[n] = np.sum(mask)/(101*101)*100

# Y_target = [int(x) for x in pd.cut(Y_target.squeeze(), bins=[0, 0.1, 100.0], include_lowest=True, labels=['0','1'])]
Y_target = [int(x) for x in pd.cut(Y_target.squeeze(), bins=[-0.1, 0.1, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], include_lowest=True, labels=['0','1','2','3','4','5','6','7','8','9','10'])]
# include_lowest=True, labels=['No salt', 'Very low', 'Low', 'Medium', 'High', 'Very high'])
# Y_target = np.expand_dims(Y_target, 1)

SaltLevel = pd.DataFrame(data={'train_ids':train_ids, 'salt_class':Y_target})

ids_train, ids_valid, SaltLevel_train, SaltLevel_valid = train_test_split(
    SaltLevel.index,
    SaltLevel,
    test_size=0.1, stratify=SaltLevel.salt_class)

sns.distplot(SaltLevel_train.salt_class, label="Train")
sns.distplot(SaltLevel_valid.salt_class, label="Valid")
plt.legend()
plt.title("Salt Class Stratified Split")
plt.savefig('salt_class.png', dpi=400)
plt.close()

data_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

class saltIDDataset(torch.utils.data.Dataset):

    def __init__(self, path_images, list_images, transforms=None, train=True):
        self.train = train
        self.path_images = path_images
        self.list_images = list_images
        self.transforms = transforms

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        image = imread(self.path_images + '/images/' + self.list_images[idx])
        if self.train:
            mask = imread(self.path_images + '/masks/' + self.list_images[idx]).astype(np.bool_).astype(np.uint8)

            if self.transforms is not None:
                transforms_input = Image.fromarray(np.dstack((image[:,:,0],image[:,:,0], mask)))
                transforms_input = self.transforms(transforms_input)
                mask = transforms_input[2:3,:,:]
                # transforms_input[-1,:,:] = transforms_input[0,:,:]
                # image = transforms_input

                image = transforms_input[0:1,:,:]

            else:
                transformTensor = transforms.ToTensor()
                image = transformTensor(Image.fromarray(image))
                mask = transformTensor(Image.fromarray(mask))

            return (image, mask)

        else:
            if self.transforms is not None:
                image = self.transforms(Image.fromarray(image))
            else:
                transformTensor = transforms.ToTensor()
                image = transformTensor(Image.fromarray(image))

            return image

class HorizontalFlip(object):
    def __call__(self, img):
        return F.hflip(img)

IMAGE_SIZE = 128
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
preprocessFlip = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    HorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
preprocess_with_augmentation = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomAffine(15),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.3,
    #                        contrast=0.3,
    #                        saturation=0.3),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

# TTA
# salt_ID_dataset_train = saltIDDataset(path_train, SaltLevel_train.train_ids.values, transforms=preprocessFlip)
# train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train, 
#                                            batch_size=16, 
#                                            shuffle=True)

salt_ID_dataset_train = saltIDDataset(path_train, SaltLevel_train.train_ids.values, transforms=preprocess_with_augmentation)
train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train, 
                                           batch_size=16, 
                                           shuffle=True,
                                           num_workers=1)

salt_ID_dataset_valid = saltIDDataset(path_train, SaltLevel_valid.train_ids.values, transforms=preprocess)
val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_valid, 
                                           batch_size=16, 
                                           shuffle=True,
                                           num_workers=1)

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)
def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

# # model = AlbuNet(num_classes=1, num_filters=32, pretrained=True, is_deconv=True)
# model = UNetResNet(152, 1, num_filters=32, dropout_2d=0.3,
#                  pretrained=True, is_deconv=True)
model = Unet()
model.train()
if torch.cuda.is_available():
    model.cuda()

epoch = 50
learning_rate = 1e-4
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for e in range(epoch):
    train_loss = []
    train_iou = []
    with tqdm(train_loader) as pbar:
        for images, masks in pbar: 
            y_pred = model(Variable(images).cuda())

            y_preds = np.squeeze((torch.sigmoid(y_pred).data.cpu().numpy() > 0.5).astype(int))
            # y_preds = np.squeeze((y_pred.data.cpu().numpy() > 0).astype(int))
            y_true = np.squeeze(masks.data.cpu().numpy())
            iou = iou_metric_batch(y_true, y_preds)
            train_iou.append(iou)

            loss = loss_fn(y_pred, Variable(masks.cuda()))
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            pbar.set_description("Loss: %.2f, IoU: %.2f, Progress" % (loss, iou))
        
    val_loss = []
    val_iou = []
    with tqdm(val_loader) as pbar:
        for images, masks in pbar:
            images = images.cuda()
            y_pred = model(Variable(images))

            y_preds = np.squeeze((torch.sigmoid(y_pred).data.cpu().numpy() > 0.5).astype(int))
            # y_preds = np.squeeze((y_pred.data.cpu().numpy() > 0).astype(int))
            y_true = np.squeeze(masks.data.cpu().numpy())
            iou = iou_metric_batch(y_true, y_preds)
            val_iou.append(iou)

            loss = loss_fn(y_pred, Variable(masks.cuda()))
            val_loss.append(loss.item())

            pbar.set_description("Loss: %.2f, IoU: %.2f, Progress" % (loss, iou))

    print("Epoch: %d, Train Loss: %.3f, Train IoU: %.3f,Val Loss: %.3f, Val IoU: %.3f" % (e, np.mean(train_loss), np.mean(train_iou), np.mean(val_loss), np.mean(val_iou)))

model.eval()
y_pred_true_pairs = []
img_list = []
with tqdm(val_loader) as pbar:
    for images, masks in pbar:
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
        else:
            images = Variable(images)
            masks = Variable(masks)

        y_preds = model(images)
        for i, img in enumerate(images):
            img_list.append(img)
            y_pred = y_preds[i] 
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().data.numpy()
            y_pred_true_pairs.append((y_pred, masks[i].cpu().data.numpy()))


plt.figure(figsize=(20,20))
for j in range(10):
    q = j+1
    plt.subplot(1,2*(1+10),q*2-1)
    plt.imshow(y_pred_true_pairs[j+10][0][0] > 0.01)
    plt.subplot(1,2*(1+10),q*2)
    plt.imshow(y_pred_true_pairs[j+10][1][0])
plt.show()

# https://www.kaggle.com/leighplt/goto-pytorch-fix-for-v0-3
thresholds = []
for threshold in np.linspace(0, 1, 21):
    
    ious = []
    for y_pred, mask in y_pred_true_pairs:
        prediction = (y_pred > threshold).astype(int)
        # iou = jaccard_similarity_score(mask.flatten(), prediction.flatten())
        iou = iou_metric(mask.flatten(), prediction.flatten())
        ious.append(iou)
        
    accuracies = np.mean([np.mean(ious > iou_threshold) for iou_threshold in np.linspace(0.5, 0.95, 10)])
    thresholds.append([threshold, accuracies])
    print('Threshold: %.2f, Metric: %.3f' % (threshold, np.mean(accuracies)))

thresholds = np.asarray(thresholds)
best_threshold = thresholds[np.min(np.where(thresholds[:,1] == np.max(thresholds[:,1]))),0]
print('threshold: {}'.format(best_threshold))

path_test = '../input/test'
test_path_images = os.path.abspath(path_test + "/images/")
test_ids = next(os.walk(test_path_images))[2]
salt_ID_dataset_test = saltIDDataset(path_test, test_ids, transforms=preprocess, train=False)
test_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_test, 
                                           batch_size=16, 
                                           shuffle=True,
                                           num_workers=1)

model.eval()
y_pred_test = []
for images in tqdm(test_loader):
    if torch.cuda.is_available():
        images = Variable(images.cuda())
    else:
        images = Variable(images)

    y_preds = model(images)
    for i, _ in enumerate(images):
        y_pred = y_preds[i] 
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.cpu().data.numpy()
        y_pred = resize(y_pred, (101, 101, 1), mode='constant', preserve_range=True)
        y_pred_test.append(y_pred)

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

test_path_images_list = glob(os.path.join(test_path_images, "*.png"))
test_file_list = [f.split('/')[-1].split('.')[0] for f in tqdm(test_path_images_list)]
submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv('submission.csv', index = False)
