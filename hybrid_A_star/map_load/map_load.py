# map load

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_name = 'map/test2.png'


def load_img(name):
    lena = mpimg.imread(name)
    return lena
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


if __name__ == '__main__':
    image = load_img(img_name)
    image_g = rgb2gray(image)
    plt.imshow(image_g, cmap='Greys_r', origin='lower')
    plt.show()

