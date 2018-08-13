import os
import cv2
import sys
import random
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from tqdm import tqdm, tnrange
from itertools import chain

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Dropout, BatchNormalization, UpSampling2D
from keras.layers.merge import concatenate
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.vis_utils import plot_model
from keras import backend as K

path = '../input'
path_train = f'{path}/train'
path_test = f'{path}/test'
imgs_train = f'{path}/train/images'
masks_train = f'{path}/train/masks'
imgs_test = f'{path}/test/images'

IMG_SIZE = 101
TGT_SIZE = 224
DPT_SIZE = 4
MAX_DEPTH = None # set after loading data

def upsample(img, img_size_target=TGT_SIZE):
    """Resize image to target"""
    img_size = img.shape[0]
    if img_size == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img, img_size_orig=IMG_SIZE):
    """Resize image to original"""
    img_size = img.shape[0]
    if img_size == img_size_orig:
        return img
    return resize(img, (img_size_orig, img_size_orig), mode='constant', preserve_range=True)

train_df = pd.read_csv(f'{path}/train.csv', index_col="id", usecols=[0])
depths_df = pd.read_csv(f'{path}/depths.csv', index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = [np.array(load_img(f'{imgs_train}/{idx}.png', color_mode="grayscale")) / 255 
                      for idx in tqdm(train_df.index)]

train_df["masks"] = [np.array(load_img(f'{masks_train}/{idx}.png', color_mode="grayscale")) / 255 
                     for idx in tqdm(train_df.index)]

MAX_DEPTH = max(train_df["z"])
train_df["depth"] = [np.ones_like(train_df.loc[i]["images"]) * train_df.loc[i]["z"] / MAX_DEPTH
                     for i in tqdm(train_df.index)]
train_df["depth"][0].shape

train_df["images_d"] = [np.dstack((train_df["images"][i], train_df["depth"][i])) for i in tqdm(train_df.index)]
train_df["images_d"][0].shape

train_df["canny"] = [cv2.threshold(cv2.Canny(np.array(load_img(f'{imgs_train}/{idx}.png', color_mode="grayscale")),100,200),90,255,cv2.THRESH_BINARY)[1] /255 
                        for idx in tqdm(train_df.index)]

train_df["median"] = [cv2.threshold(cv2.medianBlur(np.array(load_img(f'{imgs_train}/{idx}.png', color_mode="grayscale")),5),90,255,cv2.THRESH_BINARY)[1] /255 
                        for idx in tqdm(train_df.index)]

train_df["images_feat"] = [np.dstack((train_df["images"][i], train_df["canny"][i], train_df["median"][i], train_df["depth"][i])) for i in tqdm(train_df.index)]
train_df["images_feat"][0].shape

def coverage(mask):
    """Compute salt mask coverage"""
    return np.sum(mask) / (mask.shape[0]*mask.shape[1])

def coverage_class(mask):
    """Compute salt mask coverage class"""
    return (coverage(mask) * 100 //10).astype(np.int8)

def plot_imgs_masks(title,imgs, masks, preds_valid=None, thres=None, grid_width=10, zoom=1.5):
    """Visualize seismic images with their salt area mask(green) and optionally salt area prediction(pink). 
    The prediction mask can be either in probability-mask or binary-mask form(based on threshold)
    """
    grid_height = 1 + (len(imgs)-1) // grid_width
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width*zoom, grid_height*zoom))
    axes = axs.ravel()
    
    for i, img in enumerate(imgs):
        mask = masks[i]
        depth = img[0, 0, 1]
        
        
        ax = axes[i] #//grid_width, i%grid_width]
        _ = ax.imshow(img[..., 0], cmap="Greys")
        _ = ax.imshow(mask, alpha=0.3, cmap="Greens")
        
        if preds_valid is not None:
            if thres is not None:
                pred = np.array(np.round(preds_valid[i] > thres), dtype=np.float32)
            else:
                pred = preds_valid[i]
            _ = ax.imshow(pred, alpha=0.3, cmap="OrRd")
        
        _ = ax.text(2, img.shape[0]-2, depth * MAX_DEPTH//1, color="k")
        _ = ax.text(img.shape[0]-2, 2, round(coverage(mask), 2), color="k", ha="right", va="top")
        _ = ax.text(2, 2, coverage_class(mask), color="k", ha="left", va="top")
        
        _ = ax.set_yticklabels([])
        _ = ax.set_xticklabels([])
        _ = plt.axis('off')

    fig.savefig("{}.png".format(title))   # save the figure to file
    plt.close(fig)

N = 40
plot_imgs_masks("train_data",train_df.iloc[:N].images_d, train_df.iloc[:N].masks)

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, depth_train, depth_valid = train_test_split(
    train_df.index.values,
    # np.array(train_df.images_d.map(upsample).tolist()).reshape(-1, TGT_SIZE, TGT_SIZE, 2), # FJE images_d
    np.array(train_df.images_feat.map(upsample).tolist()).reshape(-1, TGT_SIZE, TGT_SIZE, 4), # FJE images_d
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, TGT_SIZE, TGT_SIZE, 1), 
    train_df.z.values,
    test_size=0.2, 
    stratify=train_df.masks.map(coverage_class), 
    random_state=1)

BATCH_SIZE = 16 
N_BATCHES = 300

# Create two instances with the same arguments
data_gen_args = dict(
#     rotation_range=2., # a bit of rotation is n
    zoom_range=[.9, 1],
    horizontal_flip=True,
    rotation_range=25.,
    )

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
image_datagen.fit(x_train, seed=1)
mask_datagen.fit(y_train, seed=1)

image_generator = image_datagen.flow(
    x_train,
    batch_size=BATCH_SIZE,
    seed=1)

mask_generator = mask_datagen.flow(
    y_train,
    batch_size=BATCH_SIZE,
    seed=1)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

X_train_l = []
Y_train_l = []
    
# add examples to list by batch
for batch_id, (x_batch, y_batch) in tqdm(enumerate(train_generator)):
    # Add full batches only - prevent odd array shapes
    if x_batch.shape[0] == BATCH_SIZE:
        X_train_l.append(x_batch)
        Y_train_l.append(y_batch)
    # Break infinite loop manually when required number of batches is reached
    if len(X_train_l) == N_BATCHES: break

# Sanity check all arrays are same shape
assert len(set(arr.shape for arr in X_train_l)) == 1
assert len(set(arr.shape for arr in Y_train_l)) == 1

# Stack list of arrays
X_train_augm = np.vstack(X_train_l)
Y_train_augm = np.vstack(Y_train_l)

# Sanity check stacking over first dimension
assert X_train_augm.shape[0] == BATCH_SIZE * N_BATCHES
assert Y_train_augm.shape[0] == BATCH_SIZE * N_BATCHES

print('Done!')

N = 60
# plot_imgs_masks(np.squeeze(X_train_augm[:N]), np.squeeze(Y_train_augm[:N]))

#Build Model
def mean_iou(y_true, y_pred, score_thres=0.5):
    """Compute mean(IoU) metric
    IoU = intersection / union
    
    For each (mask)threshold in provided range:
     - create boolean mask (from probability mask) based on threshold
     - score the mask 1 if IoU > score_threshold(0.5)
    Take the mean of the scoress

    https://www.tensorflow.org/api_docs/python/tf/metrics/mean_iou
    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_bool = tf.to_int32(y_pred > t) # boolean mask by threshold
        score, update_op = tf.metrics.mean_iou(y_true, y_pred_bool, 2) # mean score over batch(=1)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            score = tf.identity(score) #> score_thres # !! use identity to transform score in tensor
        prec.append(score) 
        
    return K.mean(K.stack(prec), axis=0)

# TODO - check performance of this metric
def dice_coef(y_true, y_pred):
    """dice_coef = 2 x intersection / (union + intersection)"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def conv_block(m, ch_dim, acti, bn, res, do=0):
    """CNN block"""
    n = Conv2D(ch_dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(ch_dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

def level_block(m, input_depth, ch_dim, depth, inc_rate, acti, do, bn, mp, up, res):
    """Recursive CNN builder"""
    if depth > 0:
        n = conv_block(m, ch_dim, acti, bn, res) # no drop-out
        m = MaxPooling2D()(n) if mp else Conv2D(ch_dim, 3, strides=2, padding='same')(n)
        m = level_block(m, input_depth, int(inc_rate*ch_dim), depth-1, inc_rate, acti, do, bn, mp, up, res)
        # Unwind recursive stack calls - with stack variables
        if up:
            # Repeat the rows and columns of the data by 2 and 2 respectively
            m = UpSampling2D()(m)
            m = Conv2D(ch_dim, 2, activation=acti, padding='same')(m)
        else:
            # Transposed convolutions are going in the opposite direction of a normal convolution
            m = Conv2DTranspose(ch_dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, ch_dim, acti, bn, res)
    else:
        # Middle conv_block
        m = conv_block(m, ch_dim, acti, bn, res, do)
        # Concat depth information in the middle layer
        m = Concatenate()([m, input_depth])
    return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
        dropout=0.5, batchnorm=False, maxpool=True, upconv=False, residual=False):
    """Returns model"""
    inputs = Input(shape=img_shape, name='img')
    # input_depth = Input(shape=(DPT_SIZE, DPT_SIZE, 1), name='depth')
    input_depth = Input(shape=(DPT_SIZE, DPT_SIZE, 1), name='depth')
    outputs = level_block(inputs, input_depth, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    outputs = Conv2D(out_ch, 1, activation='sigmoid')(outputs)
    return Model(inputs=[inputs, input_depth], outputs=outputs)

def DecoderBlock(input_m, mid_ch, out_ch):
    """CNN block"""
    n = Conv2D(mid_ch, 3, activation='relu', padding='same')(input_m)
    n = Conv2DTranspose(out_ch, (3, 3), strides=(2, 2), padding='same', output_padding=1, activation='relu')(n)

    # n = UpSampling2D()(input_m)
    # n = Conv2D(out_ch, 2, activation='relu', padding='same')(n)
    return n

# IMG_CH = 1
IMG_CH = 3
CONV_CH = 8
DEPTH = 5
D_OUT = 0.2
BN = True
UP_CONV = False
RES = True

# model = UNet((TGT_SIZE, TGT_SIZE, IMG_CH), 
#              start_ch=CONV_CH, 
#              depth=DEPTH, 
#              dropout=D_OUT,
#              batchnorm=BN, 
#              upconv=UP_CONV,
#              residual=RES)

vgg16 = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

inputs = Input(shape=img_shape, name='img')
x = vgg16.get_layer("block1_conv1")(inputs)
conv1 = vgg16.get_layer("block1_conv2")(x)

x = MaxPooling2D((2, 2)) (conv1)
x = vgg16.get_layer("block2_conv1")(x)
conv2 = vgg16.get_layer("block2_conv2")(x)

x = MaxPooling2D((2, 2)) (conv2)
x = vgg16.get_layer("block3_conv1")(x)
x = vgg16.get_layer("block3_conv2")(x)
conv3 = vgg16.get_layer("block3_conv3")(x)

x = MaxPooling2D((2, 2)) (conv3)
x = vgg16.get_layer("block4_conv1")(x)
x = vgg16.get_layer("block4_conv2")(x)
conv4 = vgg16.get_layer("block4_conv3")(x)

x = MaxPooling2D((2, 2)) (conv4)
x = vgg16.get_layer("block5_conv1")(x)
x = vgg16.get_layer("block5_conv2")(x)
conv5 = vgg16.get_layer("block5_conv3")(x)

# center
x = MaxPooling2D((2, 2)) (conv5)
center = DecoderBlock(x, num_filters * 8 * 2, num_filters * 8)
dec5 = DecoderBlock(concatenate([center, conv5]), num_filters * 8 * 2, num_filters * 8)
dec4 = DecoderBlock(concatenate([dec5, conv4]), num_filters * 8 * 2, num_filters * 8)
dec3 = DecoderBlock(concatenate([dec4, conv3]), num_filters * 4 * 2, num_filters * 2)
dec2 = DecoderBlock(concatenate([dec3, conv2]), num_filters * 2 * 2, num_filters)
dec1 = Conv2D(num_filters, 3, activation='relu', padding='same')(concatenate([dec2, conv1]))

outputs = Conv2D(1, (1, 1), activation='sigmoid') (dec1)

model = Model(inputs=[inputs], outputs=[outputs])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
# model.summary()

LR = 3e-4
optimizer = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay=LR/5)
loss = "binary_crossentropy" # or custom loss function like; 'dice_coef'
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", mean_iou])

model_name = f'TGS_salt_UNet_{IMG_CH}_{CONV_CH}_{DEPTH}_{D_OUT>0}_{BN}_{UP_CONV}_{RES}.h5'
model.summary()

EPOCHS = 30

callbacks = [
    EarlyStopping(patience=6, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint(model_name, 
                    monitor='val_loss',
                    verbose=1, 
                    save_best_only=True)]

# history = model.fit({'img': X_train_augm[..., :1], 
#                      'depth': X_train_augm[:, :DPT_SIZE, :DPT_SIZE, 1:]},
#                     Y_train_augm, 
#                     validation_data=({'img': x_valid[..., :1], 
#                                       'depth': x_valid[:, :DPT_SIZE, :DPT_SIZE, 1:]}, y_valid),
#                     batch_size=BATCH_SIZE,
#                     epochs=EPOCHS,
#                     callbacks=callbacks)

history = model.fit({'img': X_train_augm[..., :3]}, 
                    Y_train_augm, 
                    validation_data=({'img': x_valid[..., :3]}, 
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=callbacks)

fig, (ax_loss, ax_acc, ax_iou) = plt.subplots(1, 3, figsize=(15,5))

_ = ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
_ = ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
_ = ax_loss.legend()
_ = ax_loss.set_title('Loss')
_ = ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
_ = ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
_ = ax_acc.legend()
_ = ax_acc.set_title('Accuracy')
_ = ax_iou.plot(history.epoch, history.history["mean_iou"], label="Train IoU")
_ = ax_iou.plot(history.epoch, history.history["val_mean_iou"], label="Validation IoU")
_ = ax_iou.legend()
_ = ax_iou.set_title('IoU')

fig.savefig("train_progress.png")   # save the figure to file
plt.close(fig)

# Load best model
model = load_model(model_name, custom_objects={'mean_iou': mean_iou})
# Evaluate best model on validation set
model.evaluate({'img': x_valid[..., :3], 
                'depth': x_valid[:, :DPT_SIZE, :DPT_SIZE, 3:]}, y_valid, verbose=1)

# Predict on validation set
preds_valid = model.predict({'img': x_valid[..., :3], 
                             'depth': x_valid[:, :DPT_SIZE, :DPT_SIZE, 3:]}, 
                             verbose=1).reshape(-1, TGT_SIZE, TGT_SIZE)
preds_valid = np.array([downsample(x) for x in preds_valid])
y_valid_ori = np.array([train_df.loc[idx].masks for idx in ids_valid])

N = 80
imgs = train_df.loc[ids_valid[:N]].images_d
masks = train_df.loc[ids_valid[:N]].masks
preds = preds_valid[:N]
plot_imgs_masks("validation_prediction",imgs, masks, preds_valid)

def iou_metric(labels, y_pred, print_table=False):
    """
    src: https://www.kaggle.com/aglotero/another-iou-metric"""
    class_bins = 2

    # H : ndarray, shape(nx, ny)
    # The bi-dimensional histogram of samples x and y. 
    # Values in x are histogrammed along the first dimension and values in y are histogrammed along the second dimension.
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(class_bins, class_bins))[0] # was 0

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=class_bins)[0] # was 0 (0: no mask, 1: mask)
    area_pred = np.histogram(y_pred, bins=class_bins)[0] # was 0
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
    """Compute IoU batchwise"""
    batch_size = y_true_in.shape[0]
    return np.mean([iou_metric(y_true_in[b], y_pred_in[b]) for b in range(batch_size)])

thresholds = np.linspace(0.1, 0.9, 40)
ious = np.array([iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) 
                 for threshold in tqdm(thresholds)])

threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

N = 80
imgs = train_df.loc[ids_valid[:N]].images_d
masks = train_df.loc[ids_valid[:N]].masks
preds = preds_valid[:N]
plot_imgs_masks("validation_prediction_best_threshold",imgs, masks, preds_valid, threshold_best)

x_test = [upsample(np.array(load_img(f"{path_test}/images/{idx}.png", grayscale=True))) / 255 
                   for idx in tqdm(test_df.index)]

x_test = np.array(x_test).reshape(-1, TGT_SIZE, TGT_SIZE, 1)
# TODO Create depth layer

x_test_d = [np.ones((4,4,1)) * (test_df.loc[i]["z"] / MAX_DEPTH)
                     for i in tqdm(test_df.index)] 
x_test_d[0].shape
x_test_d = np.array(x_test_d).reshape(-1, DPT_SIZE, DPT_SIZE, 1)
x_test_d.shape

preds_test = model.predict({'img': x_test, 'depth': x_test_d}) 

# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', string=True):
    """Convert binary mask image to run-length array or string.
    
    pixel==1 locations are returned in format: <start length> ...
    
    Args:
    img: image in shape [n, m]
    order: is down-then-right, i.e. Fortran(F)
    string: return in string or array

    Return:
    run-length as an array or string
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

    if string:
        z = ''
        for rr in runs:
            z += f'{rr[0]} {rr[1]} '
        return z[:-1]
    else:
        return runs

pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]) > threshold_best)) 
             for i, idx in enumerate(tqdm(test_df.index.values))}

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
sub.head()