from model import *
from data import *
import cv2
import os
from skimage.transform import resize
# data_gen_args = dict(zoom_range=0.005)

myGene = trainGenerator(2,'data/parts/train','image','label',data_gen_args,save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_parts.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=1000,epochs=10,callbacks=[model_checkpoint])

img_list = os.listdir("./data/parts/test/")

model = unet()
model.load_weights("./unet_parts.hdf5")

results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/parts/test",results)
