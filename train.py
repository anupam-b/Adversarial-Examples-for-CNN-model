from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

from call_model import *

train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
    "train_augmented",
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42)

validation_generator = train_datagen.flow_from_directory(
    "train_augmented",
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42)


tmp = pd.DataFrame(columns=['ClassId', 'ModelId', 'SignName'])
csv_data = pd.read_csv("sign_name.csv")
for i, item in csv_data.iterrows():
    tmp.loc[i] = [item['ClassId'], train_generator.class_indices[str(item['ClassId'])], item['SignName']]
tmp.to_csv("sign_name.csv", sep=',', index = False)

model = build_cnn((32, 32, 3))

steps_per_epoch=train_generator.n//train_generator.batch_size
val_steps=validation_generator.n//validation_generator.batch_size+1

modelCheckpoint = ModelCheckpoint("models/weights_cnn.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=6, verbose=0, mode='auto')

callbacks_list = [modelCheckpoint, earlyStop]

history = model.fit_generator(
    train_generator,
    workers=6,
    epochs=100,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    validation_data=validation_generator,
    callbacks=callbacks_list,
    shuffle=True)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
