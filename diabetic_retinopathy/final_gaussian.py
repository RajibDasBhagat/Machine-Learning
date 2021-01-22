#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Dropout
from keras.layers import Flatten, Input, Reshape
from keras.layers import ActivityRegularization, BatchNormalization
from keras.preprocessing import image
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras import backend as K
import tensorflow as tf
K.set_image_dim_ordering('th')
#from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

import numpy as np
from PIL import Image
from PIL import ImageFile
import time
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seed
#np.random.seed(0)

ImageFile.LOAD_TRUNCATED_IMAGES = True
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

print("My GPU is as follows :"+tf.test.gpu_device_name())


# file paths

train_path = '/scratch/scratch2/_retinopathy/mtp3/4.1000_gaussian/train'
test_path  = '/scratch/scratch2/_retinopathy/mtp3/4.1000_gaussian/test'
validation_path = '/scratch/scratch2/_retinopathy/mtp3/4.1000_gaussian/validation'

print("train_path:",train_path)
print("test_path:",test_path)
print("valid_path:",validation_path)


NUM_CLASSES=5
WIDTH = 512
HEIGHT = 512
DEPTH = 3
inputShape = (DEPTH, HEIGHT, WIDTH)

EPOCHS = 100
INIT_LR = 0.003
BS = 32

#change the values as per required
batch_size   = BS
learning_rate = INIT_LR         #increase/decrease by a factor of 10
epochs        = EPOCHS
drop_out      = 0.5
class_mode    = "categorical"       #categorical/ binary/ sparse
input_size    = (512,512)
kernel_size   = (3,3)          #conv kernel size
pool_size     = (3,3)          #pooling kernel size
strides       = (2,2)          #pooling strides size
padding       = 'same'         #valid / same
color_mode    = "rgb"    #rgb / grayscale 

l2            = regularizers.l2(0.01)
#sgd           = keras.optimizers.SGD(lr=learning_rate, nesterov=True)
adam          = keras.optimizers.Adam(lr=learning_rate)
leakrelu      = keras.layers.LeakyReLU(alpha=0.0001)
random_normal = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
glorot        = keras.initializers.glorot_normal(seed=None)

bias_initializer=random_normal
optimizer=adam

def path():

    train_datagen = image. ImageDataGenerator(rotation_range=30, width_shift_range=0.1, \
                                              height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,\
                                              horizontal_flip=True, fill_mode="nearest")    
    test_datagen  = image.ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False,rescale=1.0/255.0)
    validation_datagen =image.ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rescale=1.0/255.0)

    train_batch   = train_datagen.flow_from_directory(train_path, target_size=input_size,
                                                      batch_size=batch_size, shuffle=True, 
                                                      class_mode=class_mode,color_mode=color_mode)

    validation_batch = train_datagen.flow_from_directory(validation_path ,target_size=input_size,
                                                        batch_size=batch_size, shuffle=True,
                                                        class_mode=class_mode, color_mode=color_mode)
    test_batch = train_datagen.flow_from_directory(test_path ,target_size=input_size,
                                                        batch_size=batch_size, shuffle=False,
                                                        class_mode=class_mode, color_mode=color_mode)

    return train_batch, validation_batch, test_batch


def network(train_batch,validation_batch,test_batch):
    model = Sequential()
    #layer1
    inputShape=(3,512,512)
    model.add(keras.layers.Conv2D(32, kernel_size, input_shape=inputShape, padding=padding, #activation=leaky_relu, 
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal, 
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer2
    model.add(keras.layers.Conv2D(32, kernel_size, padding=padding, #activation=leaky_relu, 
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal, 
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer3
    model.add(keras.layers.Conv2D(64, kernel_size, padding=padding, #activation=leaky_relu, 
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal, 
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer4
    model.add(keras.layers.Conv2D(64, kernel_size, padding=padding, #activation=leaky_relu, 
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal, 
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer5
    model.add(keras.layers.Conv2D(128, kernel_size, padding=padding, #activation=leaky_relu, 
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal, 
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer6
    model.add(keras.layers.Conv2D(128, kernel_size, padding=padding, #activation=leaky_relu, 
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal, 
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding) )

    #layer7
    model.add(keras.layers.Conv2D(256, kernel_size, padding=padding, #activation=leaky_relu, 
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal, 
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    
    #layer8
    model.add(keras.layers.Conv2D(256, kernel_size, padding=padding, #activation=leaky_relu, 
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal, 
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer9
    model.add(keras.layers.Conv2D(512, kernel_size, padding=padding, #activation=leaky_relu, 
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal, 
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    
    #layer10
    model.add(keras.layers.Conv2D(512, kernel_size, padding=padding, #activation=leaky_relu, 
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal, 
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))
    model.add(keras.layers.Dropout(drop_out))

    #layer11 fully connected
    #keras.layers.Flatten(data_format=None)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.Dropout(drop_out))

    #layer12 fully connected
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.LeakyReLU(alpha=0.01))

    #layer13 fully connected
    model.add(keras.layers.Dense(5,activation='softmax'))
 
    #opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    opt = keras.optimizers.SGD(lr=INIT_LR, momentum=0.0, decay=INIT_LR / EPOCHS, nesterov=True) 
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) #sgd
    print(model.summary())
    callbacks = [EarlyStopping(monitor='val_loss', patience=2), ModelCheckpoint(filepath='g2.h5', monitor='val_loss', save_best_only=True)]
    history=model.fit_generator(train_batch, steps_per_epoch=train_batch.samples/train_batch.batch_size,
                                epochs=EPOCHS, callbacks=callbacks, 
                                validation_data=validation_batch,
                                validation_steps=validation_batch.samples/validation_batch.batch_size, verbose=2,
                                shuffle=False)

    
    predict = model.predict_generator(test_batch, test_batch.samples/test_batch.batch_size, verbose=2)

    return history, predict, model



train_batch, validation_batch, test_batch = path()
history, predict, model = network(train_batch, validation_batch, test_batch)

# plotting accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training Curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('accu_g1.png')
plt.close()

#plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('loss_g1.png')
plt.close()


y_pred = np.argmax(predict, axis=1)

print('Confusion Matrix')
print(confusion_matrix(test_batch.classes, y_pred))
print('Classification Report')
label = [0,1,2,3,4]
print(classification_report(test_batch.classes, y_pred, label))
print("Completed!!")

accu=accuracy_score(test_batch.classes,y_pred)
print("Accuracy:",accu)
