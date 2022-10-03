#SCALE THE IMAGES: to apply scale_images into a data pipeline, we have to follow the MCSBP (Map, Cache, Shuffle, Batch, Prefetch) mrocedure

import tensorflow_datasets as tfds

def scale_image_pipeline(scale_images):
    #running the dataset through the scale_images preprocessing step
    ds = ds.map(scale_images)
    #cache the dataset for that batch
    ds = ds.cache()
    #shuffle it up
    ds = ds.shuffle(60000)
    #batch into 128 images per sample
    ds = ds.batch(128)
    #reduces the likelihood of bottlenecking
    ds = ds.prefetch(64)
    return ds
