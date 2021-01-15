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
K.set_image_dim_ordering('th')
#from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

import time

import matplotlib.pyplot as plt
plt.switch_backend('agg')


# file paths

train_path = '/scratch/scratch2/_retinopathy/5.data_gauss/train'
test_path  = '/scratch/scratch2/_retinopathy/5.data_gauss/test'


#change the values as per required
train_batch_size   = 32
valid_batch_size   = 32
learning_rate = 0.0001         #lower by a factor of 10
epochs        = 50
drop_out      = 0.5
class_mode    = "categorical"       #categorical/ binary/ sparse
input_size    = (512,512)
kernel_size   = (3,3)          #conv kernel size
pool_size     = (3,3)          #pooling kernel size
strides       = (2,2)          #pooling strides size
padding       = 'same'         #valid / same
color_mode    = "grayscale"    #rgb / grayscale 

l2            = regularizers.l2(0.01)
sgd           = keras.optimizers.SGD(lr=learning_rate, nesterov=True)
adam          = keras.optimizers.Adam(lr=learning_rate)
leakrelu      = keras.layers.LeakyReLU(alpha=0.0001)
random_normal = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
glorot        = keras.initializers.glorot_normal(seed=None)


print("gauss_random_normal")
print("train_batch:",train_batch_size)
print("test_batch:",valid_batch_size)
print("learning_rate:",learning_rate)
print("epochs:",epochs)
print("drop_put:",drop_out)
print("class_mode:",class_mode)
print("color_mode:",color_mode)

def path():

    train_datagen = image.ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False,
											 rescale=1.0/255.0, rotation_range=30, zoom_range=[0.2,1.2],
											 horizontal_flip=True, vertical_flip=True)
    test_datagen  = image.ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False,
											 rescale=1.0/255.0)

    train_batch   = train_datagen.flow_from_directory(train_path, target_size=input_size,
                                                      batch_size=train_batch_size, shuffle=True, 
                                                      class_mode=class_mode,color_mode=color_mode)

    validation_batch = train_datagen.flow_from_directory(test_path ,target_size=input_size,
                                                        batch_size=valid_batch_size, shuffle=False,
							class_mode=class_mode, color_mode=color_mode)
    return train_batch, validation_batch


def network(train_batch, validation_batch):
    model = Sequential()
    #layer1
    model.add(keras.layers.Conv2D(32, kernel_size, input_shape=(1,512,512), padding=padding, #activation='relu',
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal,
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer2
    model.add(keras.layers.Conv2D(32, kernel_size, padding=padding, #activation='relu',
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal,
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer3
    model.add(keras.layers.Conv2D(64, kernel_size, padding=padding, #activation='relu',
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal,
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer4
    model.add(keras.layers.Conv2D(64, kernel_size, padding=padding, #activation='relu' ,
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal,
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer5
    model.add(keras.layers.Conv2D(128, kernel_size, padding=padding, #activation='relu',
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal,
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer6
    model.add(keras.layers.Conv2D(128, kernel_size, padding=padding, #activation='relu',
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal,
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding) )

    #layer7
    model.add(keras.layers.Conv2D(256, kernel_size, padding=padding, #activation='relu',
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal,
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))

    #layer8
    model.add(keras.layers.Conv2D(256, kernel_size, padding=padding, #activation='relu',
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal,
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))

    #layer9
    model.add(keras.layers.Conv2D(512, kernel_size, padding=padding, #activation='relu',
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal,
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))

    #layer10 final_layer
    model.add(keras.layers.Conv2D(512, kernel_size, padding=padding, #activation='relu',
                                  use_bias=True, kernel_initializer=random_normal, bias_initializer=random_normal,
                                  kernel_regularizer=l2, bias_regularizer=l2))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding))
    model.add(keras.layers.Dropout(drop_out))

    #layer11 fullyconnected
    #keras.layers.Flatten(data_format=None)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.Dropout(drop_out))

    #layer12 fullyconnected
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.LeakyReLU(alpha=0.01))

    #layer13 fullyconnected 
    model.add(keras.layers.Dense(2,activation='softmax'))

    print(model.summary())

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    history=model.fit_generator(train_batch, steps_per_epoch=train_batch.samples/train_batch.batch_size,
                                epochs=epochs, validation_data=validation_batch,
                                validation_steps=validation_batch.samples/validation_batch.batch_size, verbose=2,
                                shuffle=False)

    model.save("model_final2.h5")

    predict = model.predict_generator(validation_batch, validation_batch.samples/validation_batch.batch_size, verbose=1)
    
    return history, predict


train_batch, validation_batch = path()
history, predict = network(train_batch, validation_batch)

# plotting accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('accuracy_final2.png')
plt.close()

# plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('loss_final2.png')
plt.close()


y_pred = np.argmax(predict, axis=1)

     
print('Confusion Matrix')
print(confusion_matrix(validation_batch.classes, y_pred))
print('Classification Report')
#target_names = [0,1]
print(classification_report(validation_batch.classes, y_pred))



y_pred = (y_pred > 0.6)
     
print('Confusion Matrix')
print(confusion_matrix(validation_batch.classes, y_pred))
print('Classification Report')
#target_names = [0,1]
print(classification_report(validation_batch.classes, y_pred))


print("Completed!!")
