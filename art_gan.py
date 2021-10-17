from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
#from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os


# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 100# Size vector to generate images from
NOISE_SIZE = 100# Configuration
EPOCHS = 10000 # number of iterations
BATCH_SIZE = 32
GENERATE_RES = 3
IMAGE_SIZE = 128 # rows/cols
IMAGE_CHANNELS = 3


training_data = np.load('cubism_data.npy')


def build_discriminator(image_shape):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2,
    input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))


    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))


    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))


    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))


    input_image = Input(shape=image_shape)
    validity = model(input_image)
    return Model(input_image, validity)




def build_generator(noise_size, channels):
    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation="relu",input_dim=noise_size))
    model.add(Reshape((4, 4, 256)))


    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    for i in range(GENERATE_RES):
         model.add(UpSampling2D())
         model.add(Conv2D(256, kernel_size=3, padding="same"))
         model.add(BatchNormalization(momentum=0.8))
         model.add(Activation("relu"))
         
    model.summary()
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    input = Input(shape=(noise_size,))
    generated_image = model(input)
    
    return Model(input, generated_image)





def save_images(cnt, noise):
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3),
        255, dtype=np.uint8)


    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5


    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c + IMAGE_SIZE] = generated_images[image_count] * 255
                
            image_count += 1

    output_path = 'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path, f"trained-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)




image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)


optimizer = Adam(1.5e-4, 0.5)

discriminator = build_discriminator(image_shape)
discriminator.compile(loss="binary_crossentropy",
optimizer=optimizer, metrics=["accuracy"])
generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)


random_input = Input(shape=(NOISE_SIZE,))

generated_image = generator(random_input)

discriminator.trainable = False

validity = discriminator(generated_image)


combined = Model(random_input, validity)
combined.compile(loss="binary_crossentropy",
optimizer=optimizer, metrics=["accuracy"])

y_real = np.ones((BATCH_SIZE, 1))
y_fake = np.zeros((BATCH_SIZE, 1))

fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))


cnt = 1
for epoch in range(EPOCHS):
 idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)
 x_real = training_data[idx]
 
 noise= np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
 x_fake = generator.predict(noise)
 
 discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)

 discriminator_metric_generated = discriminator.train_on_batch(
 x_fake, y_fake)
 
discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)

generator_metric = combined.train_on_batch(noise, y_real)


if epoch % SAVE_FREQ == 0:
   save_images(cnt, fixed_noise)
   cnt += 1
 
   print(f"{epoch} epoch, Discriminator accuracy: {100*  discriminator_metric[1]}, Generator accuracy: {100 * generator_metric[1]}")
  
