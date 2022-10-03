from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Con2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D


def build_generator():
    model = Sequential()

    #takes in random values and reshapes it to 7x7x28
    #begining of a generated image
    #define what our input layers is going to be
    model.add(Dense(7*7*128, input_dim=128)

    #Upsampling block 1
    model.add(UpSampling2D())
    model.add(Convi2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    #Upsampling block 2 
    model.add(UpSampling2D())
    model.add(Convi2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    #Convolutional block 1 
    model.add(Convi2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    #Convolutional block 2
    model.add(Convi2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    #Conv Layer to get to one channel
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    return model
