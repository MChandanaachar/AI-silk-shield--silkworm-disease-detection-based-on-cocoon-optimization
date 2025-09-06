import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers

import tensorflow as tf

filepath = 'efficientnetb3-silksheild-weights.h5'

print("Model Loaded Successfully")

def pred_silkworm_diseases(tomato_plant):
  print("tomato_plant==",tomato_plant)
  base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", input_shape= (128,128,3), pooling= 'max')
  model = Sequential([base_model,BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),bias_regularizer= regularizers.l1(0.006), activation= 'relu'),Dropout(rate= 0.45, seed= 123),Dense(5, activation= 'softmax')])
  model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

  model.load_weights(filepath)
  test_image = load_img(tomato_plant, target_size=(128, 128))
  print("Got Image for prediction")
  test_image = np.expand_dims(test_image, axis=0)
  result = model.predict(test_image)
  print('Raw result = ', result)
  pred = np.argmax(result, axis=1)[0]
  print("Prediction:", pred)
  for i in result:
     if 'e' in str(i):
         return "Silkworm - healthy", 'error.html'
     else:
        if pred==0:
            return "Silkworm - Flacheria Disease", 'silkworm_Flacheria.html'
        elif pred==1:
            return "Silkworm - Grasseria Disease", 'silkworm_Grasseria.html'
        elif pred==2:
            return "Silkworm - Muscardin Disease", 'silkworm_muscardin.html'
        elif pred==3:
            return "Silkworm - Pabrin Disease", 'silkworm_pabrin.html'
        elif pred==4:
            return "Silkworm - healthy", 'un_disease.html'
