from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Con2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

from "./build_generator.py" import build_generator
from "./build_discriminator.py" import build_discriminator
from "./limit_mem_growth.py" import limit_mem_growth
from "./scale_image_pipeline.py" import scale_image_pipeline
from "./visualize_data.py" import visualize_data_img, visualize_data_n
from "./scale_images.py" import scale_images

# limit memory
limit_mem_growth()

# load datasets
ds = tfds.load('fashion_mnist', split='train')

#change datatype into a python object using numpy
dataiterator = ds.as_numpy_iterator()
#getting data out of the pipeline
dataiterator.next()
# visualize_data
visualize_data_n(dataiterator)

# reload datasets
ds = tfds.load('fashion_mnist', split='train')

# scale_images in the data pipeline (image between [0, 255] to image between [0, 1])
ds_scaled = scale_image_pipeline(scale_images)
#to visualize data in a python object
dataiterator_scaled = ds_scaled.as_numpy_iterator()
#to check data size (dimensions)
#dataiterator_scaled_shape = ds_scaled.as_numpy_iterator().shape()

# build the generator
generator = build_generator()
#to get usefull informations about the generator
generator.summary()
#generate an img through random numbers
img = generator.predict(np.random.randn(4, 128, 1))
visualize_data_img(img)

#build the discriminator
discriminator = build_discriminator()
discriminator.summary()
#if we want to put (many) images as parameters
# img.shape => (4, 28, 28, 1)
discriminator.predict(img)
#if we want to work on a single image, put the index: img = img[0]
# img.shape => (28, 28, 1)
#discriminator.predict(np.expand_dims(img, 0))
