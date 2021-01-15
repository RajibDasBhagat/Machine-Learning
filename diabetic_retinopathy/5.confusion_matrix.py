from keras.models import load_model
import h5py
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
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import manifold
from sklearn import model_selection, linear_model
import numpy as np

filename = "outputs/model8.h5"
model = load_model(filename)
print(model.summary())
#print(model.get_weights())
#print(model.optimizer)

batch_size=16

#"/scratch/scratch2/_retinopathy/6.data_outer/test/"
test_path="/scratch/scratch2/_retinopathy/6.data_outer/test"
test_batch = image.ImageDataGenerator().flow_from_directory(
    test_path, color_mode="grayscale",
    target_size=(512, 512), class_mode="categorical", shuffle=False,batch_size=batch_size,
)
				

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_batch, (test_batch.samples/test_batch.batch_size), verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
#y_pred = (y_pred > 0.8)
     
print('Confusion Matrix')
print(confusion_matrix(test_batch.classes, y_pred))
print('Classification Report')
#target_names = [0,1]
print(classification_report(test_batch.classes, y_pred))


