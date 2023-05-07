#Crearemos una inteligencia artificial que es ARTISTA
#DeepDreams

#Importando Paquetes
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random 

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

base_model.summary()

#OBJETIVO
#El objetivo de esta función es seleccionar una capa e intentar maximizar la pérdida que son las activaciones generadas por la capa de interés.
#Podemos seleccionar cualquier capa que elijamos, las capas tempranas generan características simples como bordes y las capas profundas generan características más complejas como toda la cara, el coche o el árbol.
#La red de inicio tiene múltiples capas concatenadas llamadas "mixtas

names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]


#Modelo de Extraccion de Carracteristicas
deepdream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)


##PreProcesamiento de Imagen
Sample_Image = tf.keras.preprocessing.image.load_img(r'stary_night.jpg', target_size = (225, 375))

np.shape(Sample_Image)
#Normalizacion

Sample_Image = np.array(Sample_Image)/255.0
Sample_Image.shape


plt.imshow(Sample_Image)

Sample_Image.max()
Sample_Image.min()


#Corriendo Modelo pre Entrenado y Explorando las Activaciones

Sample_Image = tf.keras.preprocessing.image.img_to_array(Sample_Image)
Sample_Image.shape

Sample_Image = tf.Variable(tf.keras.applications.inception_v3.preprocess_input(Sample_Image))


Sample_Image = tf.expand_dims(Sample_Image, axis = 0)
np.shape(Sample_Image)


activations = deepdream_model(Sample_Image)

#Calculo de Funcion de Perdida
#OBJETIVO
#El objetivo de esta función es seleccionar una capa e intentar maximizar la pérdida que son las activaciones generadas por la capa de interés.
#Podemos seleccionar cualquier capa que elijamos, las capas tempranas generan características simples como bordes y las capas profundas generan características más complejas como toda la cara, el coche o el árbol.
#La red de inicio tiene múltiples capas concatenadas llamadas 'mixtas', podemos seleccionar cualquier capa o más desde mixta0 a mixta10.
#Calcularemos la pérdida que representa la suma de las activaciones de una capa dada.
#La mayoría de las veces queríamos minimizar la pérdida "error" a través de un descenso de gradiente, sin embargo, en el sueño profundo vamos a maximizar la pérdida!!
#Esto se hace perforando el "ascenso" de gradiente y no el "descenso" de gradiente.

def calc_loss(image, model):
# Función utilizada para el cálculo de pérdida
# Funciona alimentando la imagen de entrada a través de la red y generando activaciones
# Luego obtener el promedio y la suma de esos resultados

  img_batch = tf.expand_dims(image, axis=0) # Convertir a formato de lotes / batch
  layer_activations = model(img_batch) # Corriendo el Modelo
  print('VALORES DE ACTIVACION (LAYER OUTPUT) =\n', layer_activations)

  losses = [] # accumulator to hold all the losses
  for act in layer_activations:
    loss = tf.math.reduce_mean(act) # calculate mean of each activation 
    losses.append(loss)
  
  print('PERDIDAS (DE MULTIPLES CAPAS DE ACTIVACION) = ', losses)
  print('FORMA DE PERDIDA (DE MULTIPLES CAPAS DE ACTIVACION) = ', np.shape(losses))
  print('SUMA DE TODAS LAS PERDIDAS (DE TODAS LAS CAPAS SELECCIONADAS)= ', tf.reduce_sum(losses))

  return  tf.reduce_sum(losses) # Calculo de suma


#Probando Modelo
Sample_Image= tf.keras.preprocessing.image.load_img(r'stary_night.jpg', target_size = (225, 375))
Sample_Image = np.array(Sample_Image)/255.0
Sample_Image = tf.keras.preprocessing.image.img_to_array(Sample_Image)
Sample_Image = tf.Variable(tf.keras.applications.inception_v3.preprocess_input(Sample_Image))


loss = calc_loss(Sample_Image, deepdream_model)


loss

@tf.function
#Calculo de Ascenso de Gradiente
def deepdream(model, image, step_size):
    with tf.GradientTape() as tape:
      # Necesitamos gradientes relativos a las imagenes `img`
      # `GradientTape` solo mira `tf.Variable`s por defecto
      tape.watch(image)
      loss = calc_loss(image, model) # llama la funcion que calcula la perdida 

    # Calcula la gradiente de la perdiad con respecto a los pixeles de la imagen input
    # La sintaxis es la siguiente: dy_dx = g.gradient(y, x) 
    gradients = tape.gradient(loss, image)

    print('GRADIENTES =\n', gradients)
    print('FORMA DE GRADIENTES =\n', np.shape(gradients))

    # tf.math.reduce_std calcula la deviacion estandar de los elementos en todas las dimensiones del tensor
    gradients /= tf.math.reduce_std(gradients)  

    # En el ascenso de gradiente, la "pérdida" se maximiza para que la imagen de entrada "excite" cada vez más las capas.
    # Puedes actualizar la imagen añadiendo directamente los gradientes (¡porque tienen la misma forma!)
    image = image + gradients * step_size
    image = tf.clip_by_value(image, -1, 1)

    return loss, image


def run_deep_dream_simple(model, image, steps=100, step_size=0.01):
  # Convierte de uint8 al rango esperado por el modelo.
  image = tf.keras.applications.inception_v3.preprocess_input(image)

  for step in range(steps):
    loss, image = deepdream(model, image, step_size)
    
    if step % 100 == 0:
      plt.figure(figsize=(12,12))
      plt.imshow(deprocess(image))
      plt.show()
      print ("Step {}, loss {}".format(step, loss))

  plt.figure(figsize=(12,12))
  plt.imshow(deprocess(image))
  plt.show()

  return deprocess(image)

def deprocess(image):
  image = 255*(image + 1.0)/2.0
  return tf.cast(image, tf.uint8)


Sample_Image= tf.keras.preprocessing.image.load_img(r'stary_night.jpg', target_size = (225, 375))
Sample_Image = np.array(Sample_Image)
dream_img = run_deep_dream_simple(model=deepdream_model, image=Sample_Image, 
                                  steps=2000, step_size=0.001)



# puedes ejecutar el algoritmo en varios tamaños de la imagen
OCTAVE_SCALE = 1.3

Sample_Image= tf.keras.preprocessing.image.load_img(r'stary_night.jpg', target_size = (225, 375))

image = tf.constant(np.array(Sample_Image))
base_shape = tf.cast(tf.shape(image)[:-1], tf.float32)

for n in range(5):
  new_shape = tf.cast(base_shape*(OCTAVE_SCALE**n), tf.int32)
  image = tf.image.resize(image, new_shape).numpy()

  image = run_deep_dream_simple(model=deepdream_model, image=image, steps=400, step_size=0.001)


#APÉNDICE #1: CÁLCULO DE GRADIENTE Y CINTA DE GRADIENTE TF.GRADIENT()

#tf.GradientTape() se utiliza para registrar las operaciones de diferenciación automática
#Por ejemplo, supongamos que tenemos las siguientes funciones y = x^3.
#El gradiente en x = 2 puede calcularse como sigue: dy_dx = 3 * x^2 = 3 * 2^2 = 12. 

x = tf.constant(2.0)

with tf.GradientTape() as g:
  g.watch(x)
  y = x * x * x
dy_dx = g.gradient(y, x) # Se calculará hasta 12. 

dy_dx

#APÉNDICE #2: TF.FUNCTION DECORADOR
#Cuando se anota una función con tf.function, la función puede ser llamada como cualquier otra función definida por python. 
#El beneficio es que será compilado en un gráfico para que sea mucho más rápido y pueda ser ejecutado sobre TPU/GPU

# Asi era en TF 1.0 
import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.square(x)
z = tf.add(x, y)

sess = tf.Session()

z0 = sess.run([z], feed_dict={x: 2.})        # 6.0
z1 = sess.run([z], feed_dict={x: 2., y: 2.}) # 4.0


# Asi es ahora
import tensorflow as tf

@tf.function
def compute_z0(x):
  return tf.add(x, tf.square(x))

@tf.function
def compute_z1(x, y):
  return tf.add(x, y)

z0 = compute_z0(2.)
z1 = compute_z1(2., 2.)

z0

z1






















