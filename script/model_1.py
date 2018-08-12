# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
%matplotlib inline
from skimage.transform import resize

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.optimizers as optimizers
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

import os
import sys
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

depth = pd.read_csv("../input/depths.csv")
train_list = pd.read_csv("../input/train.csv")
train_depth = train_list[['id']]
train_depth = depth[depth.id.isin(train_depth.id)]#.sort_values(by=['z'])

# Set some parameters
im_width = 128
im_height = 128
im_chan = 1
path_train = '../input/train/'
path_test = '../input/test/'

ids= ['b183b2ddc4','47bd268dcd','75eaea4cdb','074673a5f1','b04c03e3d4']
# ids = ['58d177c0d0','9b29ca561d','3e84576386','c20c019650']
# ids = train_depth.loc[20:50]['id'].values
# ids = train_depth.iloc[:5,:]['id'].values
plt.figure(figsize=(20,20))
for j, img_name in enumerate(ids):
    q = j+1
    
    img = load_img('../input/train/images/' + img_name + '.png')
    plt.subplot(len(ids),5,q*5-4)
    plt.imshow(img)
    
    img_mask = load_img('../input/train/masks/' + img_name + '.png')
    plt.subplot(len(ids),5,q*5-3)
    plt.imshow(img_mask)
    
    canny = cv2.Canny(np.asarray(img),100,200)
    ret,img_th = cv2.threshold(canny,90,255,cv2.THRESH_BINARY)
    plt.subplot(len(ids),5,q*5-2)
    plt.imshow(img_th,cmap ='gray')
    
#     adapt_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6)
#     ret,img_th = cv2.threshold(adapt_img,90,255,cv2.THRESH_BINARY)
#     plt.subplot(len(ids),5,q*5-1)
#     plt.imshow(img_th,cmap ='gray')
    
    img = cv2.medianBlur(np.asarray(img),5)
    ret,img_th = cv2.threshold(img,90,255,cv2.THRESH_BINARY)
    plt.subplot(len(ids),5,q*5)
    plt.imshow(img_th,cmap ='gray')

plt.tight_layout()
plt.show()

train_depth['mask'] = train_depth['id'].apply(lambda x: np.sum(img_to_array(load_img('../input/train/masks/{}.png'.format(x)))[:,:,1])/255/(101*101)*100)
train_depth['is_salt'] = train_depth['mask'].copy()
# train_depth['no_salt'] = train_depth['mask']
train_depth['is_salt'][train_depth['mask']!=0] = 'salt'
train_depth['is_salt'][train_depth['mask']==0] = 'no_salt'
# train_depth['no_salt'][train_depth['mask']!=0] = 0
# train_depth['no_salt'][train_depth['mask']==0] = 1

train_depth['target'] = pd.cut(train_depth['mask'], bins=[0, 0.1, 10.0, 40.0, 60.0, 90.0, 100.0], 
       include_lowest=True, labels=['No salt', 'Very low', 'Low', 'Medium', 'High', 'Very high'])

train_depth = pd.concat([train_depth, pd.get_dummies(train_depth['target'])], axis=1)
train_depth = pd.concat([train_depth, pd.get_dummies(train_depth['is_salt'])], axis=1)

X_train = np.zeros((len(train_depth.id.values), 224, 224, 3), dtype=np.uint8)
Y_train = train_depth[["salt","no_salt"]].values
# Y_train = np.reshape(Y_train, (Y_train.shape[0],1,Y_train.shape[1]))

print('Getting and resizing train images and masks ... ') 
sys.stdout.flush()
for n, id_ in tqdm_notebook(enumerate(train_depth.id.values), total=len(train_depth.id.values)):
    path = path_train
    img = load_img(path + 'images/' + id_ +'.png')
    x = img_to_array(img)[:,:,1]
    x = resize(x, (224, 224, 1), mode='constant', preserve_range=True)
    x = np.dstack((x,x,x))
    X_train[n] = x
print('Done!')

# vgg19 = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
imageNet_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
x = Dense(2, activation='softmax', name='predictions')(imageNet_model.layers[-2].output)
model_saltClass = Model(input=imageNet_model.input, output=x)
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_saltClass.compile(adam, loss='binary_crossentropy', metrics=[metrics.binary_accuracy])

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('salt-classifier', verbose=1, save_best_only=True)
model_saltClass.fit(X_train, Y_train, validation_split=0.1, epochs=48, batch_size=8, verbose=1)

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)