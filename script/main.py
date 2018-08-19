import pandas as pd
import numpy as np

import os
from glob import glob
import sys
import random

from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.metrics import jaccard_similarity_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable

from torchvision import models
from unet_models import AlbuNet, UNet11, UNetVGG16, UNetResNet

im_width = 224
im_height = 224
im_chan = 3
batch_size = 16
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
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(path_train + '/images/' + id_)
    x = resize(img, (im_height, im_width, 1), mode='constant', preserve_range=True)
    X_train[n] = x
    mask = imread(path_train + '/masks/' + id_)
    Y_train[n] = resize(mask, (im_height, im_width, 1), 
                        mode='constant', 
                        preserve_range=True)

Y_target = [np.sum(x)/(im_height*im_width)*100 for x in Y_train]
Y_target = [int(x) for x in pd.cut(Y_target, bins=[0, 0.1, 10.0, 40.0, 60.0, 90.0, 100.0], include_lowest=True, labels=['0','1','2','3','4','5'])]
Y_target = np.expand_dims(Y_target, 1)
    # include_lowest=True, labels=['No salt', 'Very low', 'Low', 'Medium', 'High', 'Very high'])
print('Done!')

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

train_idxs = indices[split:]
val_idxs = indices[:split]

salt_ID_dataset_train = saltIDDataset(X_train_shaped[train_idxs], 
                                      train=True, 
                                      preprocessed_masks=Y_train_shaped[train_idxs])
salt_ID_dataset_val = saltIDDataset(X_train_shaped[val_idxs], 
                                      train=True, 
                                      preprocessed_masks=Y_train_shaped[val_idxs])
salt_ID_dataset_pretrain = saltIDDataset(X_train_shaped[train_idxs], 
                                      train=True, 
                                      preprocessed_masks=Y_target[train_idxs])
salt_ID_dataset_preval = saltIDDataset(X_train_shaped[train_idxs], 
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

start_fm = 16


#Pretraining
model_conv = models.vgg16(pretrained=True)
if torch.cuda.is_available():
    model_conv.cuda()

num_features = model_conv.classifier[6].in_features
features = list(model_conv.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 6)]) # Add our layer with 4 outputs
model_conv.classifier = nn.Sequential(*features) # Replace the model classifier

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_conv.parameters(), lr=0.001)

mean_train_losses = []
mean_val_losses = []
previous_val_losses = 1000
for epoch in range(25):
    train_losses = []
    val_losses = []
    with tqdm(pretrain_loader) as pbar:
        for images, masks in pbar:    
            if torch.cuda.is_available():    
                images = Variable(images.cuda())
                masks = Variable(masks.cuda())

            images = Variable(images)
            masks = Variable(masks)
            
            outputs = model_conv(images)
            
            loss = criterion(outputs, np.squeeze(masks))
            train_losses.append(loss.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description("Loss: {0:.2f}".format(loss))

    with tqdm(preval_loader) as pbar:
        for images, masks in pbar:
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                masks = Variable(masks.cuda())
            
            images = Variable(images)
            masks = Variable(masks)
            
            outputs = model_conv(images)
            loss = criterion(outputs, np.squeeze(masks))
            val_losses.append(loss.data)
    
    mean_train_losses.append(np.mean(train_losses))
    mean_val_losses.append(np.mean(val_losses))
    # Print Loss
    print('Epoch: {}. Train Loss: {}. Val Loss: {}'.format(epoch+1, np.mean(train_losses), np.mean(val_losses)))

    if mean_val_losses < previous_val_losses:
        previous_val_losses = mean_val_losses
        model_conv.save_state_dict('pretrained.pt')

model = UNet11(pretrained=False)
if torch.cuda.is_available():
    model.cuda();

learning_rate = 1e-3
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

mean_train_losses = []
mean_val_losses = []
for epoch in range(25):
    train_losses = []
    val_losses = []
    with tqdm(train_loader) as pbar:
        for images, masks in pbar:    
            if torch.cuda.is_available():    
                images = Variable(images.cuda())
                masks = Variable(masks.cuda())

            images = Variable(images)
            masks = Variable(masks)
            
            outputs = model(images)        
            
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
            
            images = Variable(images)
            masks = Variable(masks)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_losses.append(loss.data)
    
    mean_train_losses.append(np.mean(train_losses))
    mean_val_losses.append(np.mean(val_losses))
    # Print Loss
    print('Epoch: {}. Train Loss: {}. Val Loss: {}'.format(epoch+1, np.mean(train_losses), np.mean(val_losses)))

train_loss_series = pd.Series(mean_train_losses)
val_loss_series = pd.Series(mean_val_losses)
#train_loss_series.plot(label="train")
#val_loss_series.plot(label="validation")
#plt.legend()

y_pred_true_pairs = []
for images, masks in val_loader:
    # images = Variable(images.cuda())
    images = Variable(images)
    y_preds = model(images)
    for i, _ in enumerate(images):
        y_pred = y_preds[i] 
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.cpu().data.numpy()
        y_pred_true_pairs.append((y_pred, masks[i].numpy()))

# https://www.kaggle.com/leighplt/goto-pytorch-fix-for-v0-3
thresholds = []
for threshold in np.linspace(0, 1, 51):
    
    ious = []
    for y_pred, mask in y_pred_true_pairs:
        prediction = (y_pred > threshold).astype(int)
        iou = jaccard_similarity_score(mask.flatten(), prediction.flatten())
        ious.append(iou)
        
    accuracies = np.mean([np.mean(ious > iou_threshold) for iou_threshold in np.linspace(0.5, 0.95, 10)])
    thresholds.append([threshold, accuracies])
    print('Threshold: %.1f, Metric: %.3f' % (threshold, np.mean(accuracies)))

thresholds = np.asarray(thresholds)
best_threshold = thresholds[np.min(np.where(thresholds[:,1] == np.max(thresholds[:,1]))),0]
print('threshold: {}'.format(best_threshold))

X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.uint8)
print('Getting and resizing test images')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(path_test + '/images/' + id_)
    x = resize(img, (128, 128, 1), mode='constant', preserve_range=True)
    X_test[n] = x

X_test_shaped = X_test.reshape(-1, im_chan, im_height, im_height)/255
X_test_shaped = X_test_shaped.astype(np.float32)

salt_ID_dataset_test = saltIDDataset(X_test_shaped, 
                                      train=False) 
test_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_test, 
                                           batch_size=batch_size, 
                                           shuffle=False)

y_pred_test = []
for images in test_loader:
    if torch.cuda.is_available():
        images = Variable(images.cuda())
    else:
        images = Variable(images)

    y_preds = model(images)
    for i, _ in enumerate(tqdm(images)):
        y_pred = y_preds[i] 
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.cpu().data.numpy()
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

test_file_list = [f.split('/')[-1].split('.')[0] for f in test_path_images_list]
submit = pd.DataFrame([test_file_list, all_masks]).T
submit.columns = ['id', 'rle_mask']
submit.to_csv('submission.csv', index = False)
