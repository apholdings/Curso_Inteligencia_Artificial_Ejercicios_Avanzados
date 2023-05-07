import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Unzip Archivo
dataset_path = "./cats_and_dogs_filtered.zip"
zip_object = zipfile.ZipFile(file=dataset_path, mode="r")
zip_object.extractall("./")
zip_object.close()

#Path al set de datos
dataset_path_new = "./cats_and_dogs_filtered"

train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")

#ARMANDO EL MODELO
IMG_SHAPE = (128,128,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top = False, weights="imagenet")

#base_model.summary()


#Congelado de Modelo Base
base_model.trainable = True

#DEFINICION DE LAA CABEZA
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

#Definir capa output
prediction_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)


#DEFINICION MODELO
model= tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)


#COMPILADO
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])


#Creando Data Generators
data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)


train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode="binary")
valid_generator = data_gen_valid.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode="binary")


#ENTRENANDO MODELO
model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

# Afinamiento
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at:]:
    layer.trainable = False

# Compilado
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Afinamiento / Entrenamiento
model.fit_generator(train_generator,  
                    epochs=5, 
                    validation_data=valid_generator)

# Evaluacion
valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)



















