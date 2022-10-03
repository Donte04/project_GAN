#LIMIT THE GPU CONSUMPTIONS

#Bring the tensorflow
import tensorflow as tf

def limit_mem_growth:
    #grab all the GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    #loop through them
    for gpu in gpus:
        #limit memory growth
        tf.config.experimental.set_memory_growth(gpu, True)
