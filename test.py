# import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import visualkeras

from call_model import *

model = load_model_weights("models/weights_cnn.hdf5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "gtsrb-german-traffic-sign/Testing",
    target_size=(32, 32),
    batch_size=32,
    class_mode=None,
    shuffle=False)

val_steps = test_generator.n//test_generator.batch_size+1

preds = model.predict_generator(test_generator, verbose=1, steps=val_steps)
Ypred = np.argmax(preds, axis=1)

dataframe = pd.read_csv("sign_name.csv")
l = []
for i in range(len(Ypred)):
    l.append(int(dataframe['ClassId'].loc[dataframe['ModelId'] == Ypred[i]]))
result = np.array(l)

test = pd.read_csv("gtsrb-german-traffic-sign/Test.csv")
count_err = 0
for i, item in test.iterrows():
	if (item['ClassId'] != result[i]):
		count_err += 1

accuracy = round((1 - (float(count_err) / float(len(Ypred)))) * 100, 2)
print("Accuracy "+str(accuracy)+"%")

keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True
)

visualkeras.layered_view(model, type_ignore=[keras.layers.Flatten], legend=True,to_file='architecture.png')
