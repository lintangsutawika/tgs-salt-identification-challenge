import pandas as pd
import numpy as np

import gc
import os
from glob import glob
import sys
import random

from tqdm import tqdm,tqdm_notebook
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable

from torchvision import models
from unet_models import AlbuNet, UNet11, UNetVGG16, UNetResNet, Unet

im_width = 128
im_height = 128
im_chan = 3
batch_size = 32
path_train = '../input/train'
path_test = '../input/test'

train_path_images = os.path.abspath(path_train + "/images/")
train_path_masks = os.path.abspath(path_train + "/masks/")

test_path_images = os.path.abspath(path_test + "/images/")
test_path_masks = os.path.abspath(path_test + "/masks/")

train_path_images_list = glob(os.path.join(train_path_images, "*.png"))
train_path_masks_list = glob(os.path.join(train_path_masks, "*.png"))
test_path_images_list = glob(os.path.join(test_path_images, "*.png"))
test_path_masks_list = glob(os.path.join(test_path_masks, "*.png"))

train_ids = next(os.walk(train_path_images))[2]
test_ids = next(os.walk(test_path_images))[2]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool_)
Y_target = np.zeros((len(train_ids), 1), dtype=int)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(path_train + '/images/' + id_)
    x = resize(img, (im_height, im_width, im_chan), mode='constant', preserve_range=True)

    border = 5
    x_center_mean = x[border:-border, border:-border].mean()
    x_csum = (np.float32(x)-x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # x = std * x + mean

    # x = np.clip(x, 0, 1)
    X_train[n] = x
    # X_train[n] = np.dstack((x[:,:,0],x[:,:,1],x_csum))
    mask = imread(path_train + '/masks/' + id_)
    Y_train[n] = resize(mask, (im_height, im_width, 1), 
                        mode='constant', 
                        preserve_range=True)
    # if np.sum(Y_train[n])/(im_height*im_width)*100 == 0:
    #     Y_target[n] = 0.0
    # else:
    #     Y_target[n] = 1.0
    Y_target[n] = np.sum(Y_train[n])/(im_height*im_width)*100

# Y_target = [int(x) for x in pd.cut(Y_target.squeeze(), bins=[0, 0.1, 100.0], include_lowest=True, labels=['0','1'])]
Y_target = [int(x) for x in pd.cut(Y_target.squeeze(), bins=[-0.1, 0.1, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], include_lowest=True, labels=['0','1','2','3','4','5','6','7','8','9','10'])]
# include_lowest=True, labels=['No salt', 'Very low', 'Low', 'Medium', 'High', 'Very high'])
Y_target = np.expand_dims(Y_target, 1)

X_train = np.append(X_train, [np.fliplr(x) for x in X_train],axis=0)
Y_train = np.append(Y_train, [np.fliplr(x) for x in Y_train],axis=0)
Y_target = np.append(Y_target, Y_target,axis=0)

print('Done!')

# src: https://www.kaggle.com/aglotero/another-iou-metric
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

# https://stackoverflow.com/questions/50052295/how-do-you-load-images-into-pytorch-dataloader
class saltIDDataset(torch.utils.data.Dataset):

    def __init__(self,preprocessed_images,train=True, preprocessed_masks=None):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.train = train
        self.images = preprocessed_images
        if self.train:
            self.masks = preprocessed_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = None
        if self.train:
            mask = self.masks[idx]
            return (image, mask)
        else:
            return image

X_train_shaped = X_train.reshape(-1, im_chan, im_height, im_width)/255
Y_train_shaped = Y_train.reshape(-1, 1, im_height, im_width)

X_train_shaped = X_train_shaped.astype(np.float32)
Y_train_shaped = Y_train_shaped.astype(np.float32)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(4200)
    np.random.seed(133700)

indices = list(range(len(X_train_shaped)))
np.random.shuffle(indices)

val_size = 1/10
split = np.int_(np.floor(val_size * len(X_train_shaped)))

ids_train, ids_valid, x_train, x_valid, y_train, y_valid = train_test_split(
    indices,
    X_train_shaped,
    Y_train_shaped, 
    test_size=0.1, stratify=Y_target, random_state=1337)

train_idxs = ids_train
val_idxs = ids_valid
# train_idxs = indices[split:]
# val_idxs = indices[:split]

salt_ID_dataset_train = saltIDDataset(X_train_shaped[train_idxs], 
                                      train=True, 
                                      preprocessed_masks=Y_train_shaped[train_idxs])
salt_ID_dataset_val = saltIDDataset(X_train_shaped[val_idxs], 
                                      train=True, 
                                      preprocessed_masks=Y_train_shaped[val_idxs])
salt_ID_dataset_pretrain = saltIDDataset(X_train_shaped[train_idxs], 
                                      train=True, 
                                      preprocessed_masks=Y_target[train_idxs])
salt_ID_dataset_preval = saltIDDataset(X_train_shaped[val_idxs], 
                                      train=True, 
                                      preprocessed_masks=Y_target[val_idxs])

train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train, 
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_val, 
                                           batch_size=batch_size, 
                                           shuffle=False)

pretrain_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_pretrain, 
                                           batch_size=batch_size, 
                                           shuffle=True)

preval_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_preval, 
                                           batch_size=batch_size, 
                                           shuffle=False)


#Pretraining
model_conv = models.resnet34(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False

model_conv.fc = nn.Linear(model_conv.fc.in_features, 11)

# model_conv = models.vgg16(pretrained=True)
# num_features = model_conv.classifier[6].in_features
# features = list(model_conv.classifier.children())[:-1] # Remove last layer
# features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
# model_conv.classifier = nn.Sequential(*features) # Replace the model classifier

if torch.cuda.is_available():
    model_conv.cuda()

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_conv.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model_conv.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

mean_train_losses = []
mean_val_losses = []
previous_val_losses = 1000
for epoch in range(40):
    train_losses = []
    val_losses = []
    model_conv.train()
    with tqdm(pretrain_loader) as pbar:
        for images, masks in pbar:    
            if torch.cuda.is_available():    
                images = Variable(images.cuda())
                masks = Variable(masks.cuda())
            else:
                images = Variable(images)
                masks = Variable(masks)

            outputs = model_conv(images)
            
            loss = criterion(outputs, masks.view(-1))
            train_losses.append(loss.data.cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description("Loss: {0:.2f}".format(loss.cpu().data[0]))

    model_conv.eval()
    with tqdm(preval_loader) as pbar:
        for images, masks in pbar:
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                masks = Variable(masks.cuda())
            else:
                images = Variable(images)
                masks = Variable(masks)
            
            outputs = model_conv(images)
            loss = criterion(outputs, masks.view(-1))
            val_losses.append(loss.data)
    
    mean_train_losses.append(np.mean(train_losses))
    mean_val_losses.append(np.mean(val_losses))
    
    # Print Loss
    print('Epoch: {}. Train Loss: {}. Val Loss: {}'.format(epoch+1, np.mean(train_losses), np.mean(val_losses)))

    if mean_train_losses[0] < previous_val_losses:
        previous_val_losses = mean_train_losses[0]
        torch.save(model_conv.state_dict(), 'custom_pretrained_RESNET34.pth')

# model = Unet()
model = unet11()
# model = UNetVGG16(pretrained='custom')
# model = UNetResNet(152, 1, pretrained=True)
# model = UNetResNet(101, 1, pretrained=True, dropout_2d=0.5, is_deconv=True)
# model = UNetResNet(34, 1, pretrained=True, dropout_2d=0.5, is_deconv=True)
if torch.cuda.is_available():
    model.cuda()

learning_rate = 1e-3
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

mean_train_losses = []
mean_val_losses = []
mean_mean_iou = []
previous_val_losses = 1000
for epoch in range(5):
    train_losses = []
    train_iou = []
    val_losses = []
    val_iou = []
    with tqdm(train_loader) as pbar:
        for images, masks in pbar:    
            if torch.cuda.is_available():    
                images = Variable(images.cuda())
                masks = Variable(masks.cuda())
            else:
                images = Variable(images)
                masks = Variable(masks)
            
            outputs = model(images)        
            
            # y_preds = np.squeeze((torch.sigmoid(outputs).data.cpu().numpy() > 0.5).astype(int))
            y_preds = np.squeeze((outputs.data.cpu().numpy() > 0).astype(int))
            y_true = np.squeeze(masks.data.cpu().numpy())
            train_iou.append(iou_metric_batch(y_true, y_preds))

            loss = criterion(outputs, masks)
            train_losses.append(loss.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description("Loss: {0:.2f}".format(loss))

    with tqdm(val_loader) as pbar:
        for images, masks in pbar:
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                masks = Variable(masks.cuda())
            else:
                images = Variable(images)
                masks = Variable(masks)
            
            outputs = model(images)

            # y_preds = np.squeeze((torch.sigmoid(outputs).data.cpu().numpy() > 0.5).astype(int))
            y_preds = np.squeeze((outputs.data.cpu().numpy() > 0).astype(int))
            y_true = np.squeeze(masks.data.cpu().numpy())
            val_iou.append(iou_metric_batch(y_true, y_preds))

            loss = criterion(outputs, masks)
            val_losses.append(loss.data)
    
    mean_train_losses.append(np.mean(train_losses))
    mean_val_losses.append(np.mean(val_losses))
    mean_mean_iou.append(np.mean(val_iou))
    # Print Loss
    print('Epoch: {}. Train Loss: {}. Train IoU: {}. Val Loss: {}. Val IoU: {}'.format(epoch+1, np.mean(train_losses), np.mean(train_iou), np.mean(val_losses), np.mean(val_iou)))
    if mean_val_losses[0] < previous_val_losses:
        previous_val_losses = mean_val_losses[0]
        torch.save(model.state_dict(), 'UNET_RESNET34.pth')

# train_loss_series = pd.Series(mean_train_losses)
# val_loss_series = pd.Series(mean_val_losses)
#train_loss_series.plot(label="train")
#val_loss_series.plot(label="validation")
#plt.legend()

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


plt.figure(figsize=(20,10))
for j, img in enumerate(img_list):
    q = j+1
    plt.subplot(1,2*(1+len(ids)),q*2-1)
    plt.imshow(y_pred_true_pairs[j,0] > 0.4)
    plt.subplot(1,2*(1+len(ids)),q*2)
    plt.imshow(y_pred_true_pairs[j,1])
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

X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.uint8)
print('Getting and resizing test images')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(path_test + '/images/' + id_)
    x = resize(img, (im_height, im_width, 1), mode='constant', preserve_range=True)
    X_test[n] = x

X_test_shaped = X_test.reshape(-1, im_chan, im_height, im_width )/255
X_test_shaped = X_test_shaped.astype(np.float32)

salt_ID_dataset_test = saltIDDataset(X_test_shaped, 
                                      train=False) 
test_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_test, 
                                           batch_size=batch_size, 
                                           shuffle=False)

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

# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

all_masks = []
for p_mask in tqdm(list(binary_prediction)):
    p_mask = RLenc(p_mask)
    # all_masks.append(' '.join(map(str, p_mask)))
    all_masks.append(''.join(map(str, p_mask)))


test_file_list = [f.split('/')[-1].split('.')[0] for f in tqdm(test_path_images_list)]
submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv('submission.csv', index = False)
