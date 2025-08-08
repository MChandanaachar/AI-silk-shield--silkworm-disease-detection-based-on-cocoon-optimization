import tensorflow.keras.applications.EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import os
IMAGE_SIZE = [128,128]
#from keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_acc", patience=3, verbose=1, factor=0.3, min_lr=0.0000001
)
early_stop = EarlyStopping(
    patience=10,
    verbose=1,
    monitor="val_acc",
    mode="max",
    min_delta=0.001,
    restore_best_weights=True,
)
vgg16=EfficientNetB3(input_shape = IMAGE_SIZE + [3], weights='imagenet', include_top=False)
x1= Flatten()(vgg16.output)
prediction1 = Dense(5, activation='softmax')(x1)
model2 = Model(inputs = vgg16.inputs, outputs = prediction1)
model2.summary()
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./Dataset/Train', # relative path from working directoy
                                                 target_size = (128, 128),
                                                 batch_size = 6, class_mode = 'categorical')
valid_set = test_datagen.flow_from_directory('./Dataset/Test', # relative path from working directoy
                                             target_size = (128, 128), 
                                        batch_size = 3, class_mode = 'categorical')

labels = (training_set.class_indices)
print(labels)
model2.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=["acc"])
hist2 = model2.fit(training_set, validation_data=valid_set, epochs=20, steps_per_epoch=len(training_set), validation_steps=len(valid_set),callbacks=[learning_rate_reduction, early_stop])
import matplotlib.pyplot as plt

x=hist2
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(x.history['loss'], label='Training Loss')
plt.plot(x.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(x.history['acc'], label='Training Accuracy')
plt.plot(x.history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()