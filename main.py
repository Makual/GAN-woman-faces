!pip install -q kaggle

!rm -r ~/.kaggle
!mkdir ~/.kaggle
!mv ./kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets list

!kaggle datasets download -d arnaud58/selfie2anime

import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, Reshape, Conv2D, Conv2DTranspose, ReLU, LeakyReLU,InputLayer, UpSampling2D,ZeroPadding2D
import tensorflow.keras.datasets as dts 
from PIL import Image, ImageOps

from tensorflow.compat.v1 import keras
import tensorflow as tf

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)

import zipfile
with zipfile.ZipFile('/content/selfie2anime.zip', 'r') as zip_ref:
  zip_ref.extractall('/content/faceTOanime_2')

from skimage.io import imread
import glob

imgs = []
for filename in glob.glob('/content/faceTOanime_2/trainA/*.jpg'):
  imgs.append(imread(filename))

for i in range(len(imgs)):
  imgs[i] = np.array(Image.fromarray(imgs[i]).resize((128,128)))

imgs = np.array(imgs)

x_train = imgs
x_train = (x_train.astype(np.float32) - 127.5) / 127.5

x_train.shape

lr = 0.00005
batch_size = 64
betas = (0.5, 0.999)
noise_size = 256

discriminator = keras.Sequential([
  InputLayer((128,128,3)),
  Conv2D(32, kernel_size=3, strides=2, padding="same"),
  LeakyReLU(alpha=0.2),
  Dropout(0.25),
  Conv2D(64, kernel_size=3, strides=2, padding="same"),
  ZeroPadding2D(padding=((0,1),(0,1))),
  BatchNormalization(momentum=0.8,),
  LeakyReLU(alpha=0.2),
  Dropout(0.25),
  Conv2D(128, kernel_size=3, strides=2, padding="same"),
  BatchNormalization(momentum=0.8),
  LeakyReLU(alpha=0.2),
  Dropout(0.25),
  Conv2D(256, kernel_size=3, strides=1, padding="same"),
  BatchNormalization(momentum=0.8),
  LeakyReLU(alpha=0.2),
  Dropout(0.25),
  Flatten(),
  Dense(1, activation='sigmoid'),
])

discriminator.summary()

optimizer = keras.optimizers.Adam(lr,betas[0],betas[1])
discriminator.compile(loss='binary_crossentropy', 
  optimizer=optimizer)

generator = keras.Sequential([
  InputLayer((noise_size,)),
  Dense(128 * 8 * 8, activation="relu"),
  Reshape((8, 8, 128)),
  UpSampling2D(),
  Conv2D(128, kernel_size=3, padding="same"),
  BatchNormalization(momentum=0.8),
  ReLU(),
  UpSampling2D(),
  Conv2D(128, kernel_size=3, padding="same"),
  BatchNormalization(momentum=0.8),
  ReLU(),
  UpSampling2D(),
  Conv2D(128, kernel_size=3, padding="same"),
  BatchNormalization(momentum=0.8),
  ReLU(),
  UpSampling2D(),
  Conv2D(128, kernel_size=3, padding="same"),
  BatchNormalization(momentum=0.8),
  ReLU(),
  Conv2D(128, kernel_size=3, padding="same"),
  BatchNormalization(momentum=0.8),
  ReLU(),
  Conv2D(3, kernel_size=3, padding="same", activation='tanh')
])

generator.summary()

optimizer = keras.optimizers.Adam(lr,betas[0],betas[1])
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

optimizer = keras.optimizers.Adam(lr,betas[0],betas[1])
z = Input(shape=(noise_size,))
img = generator(z)
valid = discriminator(img)
combined = keras.Model(z, valid)

combined.compile(loss='binary_crossentropy', optimizer=optimizer)

!mkdir gan
!mkdir gan/images


def save_imgs(epoch):
  r, c = 5, 5

  noise = np.random.normal(0, 1, (25, noise_size))
  generated_samples = generator(noise)
  generated_samples = np.array(generated_samples)
  generated_samples = ((generated_samples * 0.5) + 0.5) * 255
  generated_samples = generated_samples.round()
  generated_samples = generated_samples.astype(np.uint8)

  fig, axs = plt.subplots(r, c)
  cnt = 0
  for i in range(r):
    for j in range(c):
      axs[i,j].imshow(generated_samples[cnt])
      #axs[i,j].axis('off')
      cnt += 1
  fig.savefig("gan/images/mnist_%d.png" % epoch)
  plt.close()

#discriminator = keras.models.load_model('/content/drive/MyDrive/ColabFiles/discriminator_4000.h5')
#generator = keras.models.load_model('/content/drive/MyDrive/ColabFiles/generator_4000.h5')

num_epochs = 7000
save_step = 100

for epoch in range(0,num_epochs):  
  idx = np.random.randint(0,x_train.shape[0], batch_size)
  imgs = x_train[idx]
  noise = np.random.normal(0, 1, (batch_size, noise_size))
  gen_imgs = generator.predict(noise)

  d_loss_real = discriminator.train_on_batch(imgs, np.ones((batch_size, 1)))
  d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
  d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

  noise = np.random.normal(0, 1, (batch_size, noise_size))
  valid_y = np.array([1] * batch_size)

  discriminator.trainable = False
  g_loss = combined.train_on_batch(noise, valid_y)
  discriminator.trainable = True

  if (epoch % 100) == 0:
    save_imgs(epoch)
  
  print (epoch, d_loss, g_loss)

noise = np.random.normal(0, 1, (16, noise_size))
generated_samples = generator(noise)
generated_samples = np.array(generated_samples)
generated_samples = ((generated_samples * 0.5) + 0.5) * 255
generated_samples = generated_samples.round()
generated_samples = generated_samples.astype(np.uint8)

generated_samples.shape

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i])
    plt.xticks([])
    plt.yticks([])

plt.imshow(generated_samples[9])

'''discriminator.save('discriminator_6000.h5')
generator.save('generator_6000.h5')

import shutil
shutil.make_archive('history', 'zip', '/content/gan/images')'''
