#VISUALIZE DATA

#do some data transformation
import numpy as np
from matplotlib import pyplot as plt

def visualize_data_n(dataiterator)
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, in range(4):
        #grap on image and label
        sample = dataiterator.next()
        #plot the image using a specific subplot
        ax[idx].imshow(np.squeeze(sample['image']))
        #appending the image label as the plot title
        ax[idx].title.set_text(sample['label'])

def visualize_data_img(img):
    #setup the subplot formatting
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(img):
        ax[idx].imshow(np.squeeze(img))
        ax[idx].title.set_text(idx)
