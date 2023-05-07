

    #Un RNN contiene un bucle temporal en el que la capa oculta no sólo da una salida sino que se alimenta a sí misma también.

    #Se añade una dimensión extra que es el tiempo#

    #RNN puede recordar lo que pasó en la anterior marca de tiempo, así que funciona muy bien con la secuencia de texto.

    #Los RNA de avance están tan limitados por su número fijo de entradas y salidas.

    #Por ejemplo, una CNN tendrá una imagen de tamaño fijo (28x28) y genera una salida fija (clase o probabilidades).

    #Las RNA de avance tienen una configuración fija, es decir: el mismo número de capas y pesos ocultos.

    #Las redes neuronales recurrentes ofrecen una gran ventaja sobre las RNA de avance y son mucho más divertidas!

    #RNN nos permiten trabajar con una secuencia de vectores:
        #Secuencia en las entradas
        #Secuencia en las salidas
        #Secuencia en ambos!

    #Las redes LSTM son un tipo de RNN que están diseñadas para recordar las dependencias a largo plazo por defecto.

    #LSTM puede recordar y recordar información por un período prolongado de tiempo.

    #NOTA: Este código es adoptado de la documentación de TF2.0: https://www.tensorflow.org/beta/tutorials/text/text_generation

    # En este tutorial, usaremos un conjunto de datos extraídos de la escritura de Shakespeare.
    #Consulta esta referencia: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    #El objetivo es entrenar a una red LSTM para predecir el próximo personaje en una secuencia de caracteres.


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
import os
import time

data_url = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


dataset_text = open(data_url, 'rb').read().decode(encoding='utf-8')
print(dataset_text)

len(dataset_text)

vocab = sorted(set(dataset_text))
print ('{} unique characters'.format(len(vocab)))


# Creando un mapa de caracteres únicos a índices
char2idx = {char:index for index, char in enumerate(vocab)}
char2idx


idx2char = np.array(vocab)
idx2char

#Conversion de caracteres a numeros enteros
text_as_int = np.array([char2idx[char] for char in dataset_text])
text_as_int



print ('{} ---- caracteres mapeados a enteros (int) ---- > {}'.format(repr(dataset_text[:13]), text_as_int[:13]))


#Creando entrenamiento y lotes

    #Dividiremos el conjunto de datos en un cuadrado de caracteres con longitud de secuencia.
    #La salida será la misma que la entrada pero desplazada por un carácter
    #Ejemplo: si nuestro texto es "Hello" y seq_len = 4
        #Entrada: "Hell"
        #Salida: "ello"


# Calcula el número de ejemplos por Epoch asumiendo una secuencia de 100 caracteres 
seq_length = 100
examples_per_epoch = len(dataset_text)//seq_length
examples_per_epoch

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


for i in char_dataset.take(200):
  print(idx2char[i.numpy()])


#El método de lotes nos permite convertir fácilmente estos caracteres individuales en secuencias del tamaño deseado.
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(50):
  print(repr(''.join(idx2char[item.numpy()])))

# Para cada secuencia, duplicarla y desplazarla para formar el texto de entrada y de destino usando el método "map" para aplicar una función simple a cada lote:
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


#Imprime los primeros ejemplos de valores de entrada y de destino:
for input_example, target_example in  dataset.take(10):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))


# Shuffle the dataset and it into batches
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#Creando y Entrenando el Modelo

#Utilice tf.keras.Sequential para definir el modelo. Se utilizan tres capas:
    #tf.keras.layers.Embedding: La capa de entrada. Una tabla de búsqueda entrenable que mapeará los números de cada personaje a un vector con dimensiones embedding_dim
    #tf.keras.layers.LSTM
    #tf.keras.layers.Dense: La capa de salida, #con salidas de tamaño_vocabulario

len(vocab)

# Longitud del vocabulario en caracteres
vocab_size = len(vocab)

#Dimension de Embedding
embedding_dim = 256


#RNR Neuronas / Unidades
rnn_units = 1024

#Armando modelo
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model


model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


for input_example_batch, target_example_batch in dataset.take(10):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


model.summary()

#Para obtener predicciones reales del modelo necesitamos tomar muestras de la distribución de salida, para obtener índices de caracteres reales.

#Esta distribución se define por los logits sobre el vocabulario de caracteres.

#Nota: Es importante tomar muestras de esta distribución ya que tomar el argmax de la distribución puede hacer que el modelo se atasque fácilmente en un bucle.

#Pruébalo para el primer ejemplo del lote:
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

#Viendo los TimeSteps
sampled_indices


#Los resultados de un modelo no entrenado 
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


#Entrenamiento
#En este punto el problema puede ser tratado como un problema de clasificación estándar. 
#Dado el estado RNR anterior, y la entrada de este paso de tiempo, predecir la clase del siguiente personaje.


#Agregando un Optimizador y funcion dee perdida
#Debido a que nuestro modelo devuelve logits, necesitamos establecer el flag de from_logits.

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#Ejecutar Entrenamiento

EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


#Generar Texto

#Restaurar el último checkpoint
#Para mantener este paso de predicción simple, usa un tamaño de lote de 1.
#Por la forma en que el estado RNR se pasa de un paso a otro, el modelo sólo acepta un tamaño de lote fijo una vez construido.
#Para ejecutar el modelo con un tamaño_de_lote diferente, necesitamos reconstruir el modelo y restaurar los pesos del punto de control.


#--------------- ENTRENAMIENTO ------------------#
#tf.train.latest_checkpoint(checkpoint_dir)
#--------------- ENTRENAMIENTO ------------------#

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

#El Ciclo de predicción
#El siguiente bloque de código genera el texto:
   # Comienza eligiendo una cadena de inicio, inicializando el estado RNR y estableciendo el número de caracteres a generar.
   # Obtiene la distribución de la predicción del siguiente carácter usando la cadena de inicio y el estado RNR.
   # Luego, usa una distribución categórica para calcular el índice del carácter predicho. Usar este carácter predicho como nuestra próxima entrada al modelo.
   # El estado RNR devuelto por el modelo se retroalimenta en el modelo de modo que ahora tiene más contexto, en lugar de una sola palabra. Después de predecir la siguiente palabra, los estados RNR modificados se vuelven a introducir en el modelo, que es como aprende a medida que obtiene más contexto de las palabras predichas anteriormente.

#Para generar el texto, la salida del modelo se retroalimenta a la entrada
#Mirando el texto generado, verás que el modelo sabe cuando capitalizar, hacer párrafos e imita un vocabulario de escritura similar al de Shakespeare. Con el pequeño número de épocas de entrenamiento, aún no ha aprendido a formar oraciones coherentes.

# ------------ CREAR POEMA ----------------#

def generate_text(model, start_string):
  # Paso de evaluación (generación de texto utilizando el modelo aprendido)

  # Número de caracteres a generar
  num_generate = 1000

  # Convirtiendo nuestra cadena de inicio en números (vectorización)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Cuerda vacía para almacenar nuestros resultados
  text_generated = []

  # Las bajas temperaturas dan como resultado un texto más predecible.
  # Las temperaturas más altas dan como resultado un texto más sorprendente.
  # Experimente para encontrar el mejor escenario.
  temperature = 1.0

  # El tamano del lote es == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remueve la dimension del lote
      predictions = tf.squeeze(predictions, 0)

      # usando una distribución categórica para predecir la palabra devuelta por el modelo
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # Pasamos la palabra predicha como la siguiente entrada al modelo
      # junto con el estado oculto anterior
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


##Iimprimiendo Poema de Shakespeare
print(generate_text(model, start_string=u"ROMEO: "))

#Lo más fácil que puedes hacer para mejorar los resultados es entrenarlo por más tiempo (prueba EPOCHS=30).

#También puedes experimentar con una cadena de inicio diferente, o intentar añadir otra capa RNR para mejorar la precisión del modelo, o ajustar el parámetro de temperatura para generar predicciones más o menos aleatorias.



