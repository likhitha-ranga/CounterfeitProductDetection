from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Flatten

from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121

IMAGE_SIZE=224
BATCH_SIZE=32
CHANNELS=3
EPOCHS=17

import tensorflow as tf
dataset=tf.keras.preprocessing.image_dataset_from_directory('archive',shuffle=True,image_size=(IMAGE_SIZE,IMAGE_SIZE),batch_size=BATCH_SIZE)

def get_dataset_partitions_tf(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
  ds_size=len(ds)

  if shuffle:
    ds=ds.shuffle(shuffle_size,seed=10)
  train_size=int(train_split*ds_size)
  val_size=int(val_split*ds_size)

  train_ds=ds.take(train_size)
  val_ds=ds.skip(train_size).take(val_size)
  test_ds=ds.skip(train_size).skip(val_size)

  return train_ds,val_ds,test_ds

train_ds,val_ds,test_ds=get_dataset_partitions_tf(dataset)

train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

model=DenseNet121(include_top=False, input_shape=(224,224,3))

for layer in model.layers[:-7]:
  layer.trainable=False

flatten_layer = layers.Flatten()(model.output)


flattened_fc_layer = layers.Dense(512, activation='relu')(flatten_layer)


flattened_fc_softmax_layer = layers.Dense(2, activation='softmax')(flattened_fc_layer)

model = Model(inputs=model.inputs, outputs=flattened_fc_softmax_layer)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history=model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
    )

scores = model.evaluate(test_ds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print("Training accuracy:", acc)
print("Validation accuracy:", val_acc)
print("Training loss:", loss)
print("Validation loss:", val_loss)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Training and validation loss')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Training and validation accuracy')
plt.xlabel('epoch')
plt.show()

model.save('fakeProduct.h5')