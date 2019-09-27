# map load

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_name = 'map/empty_test.png'
resX = 0.1
resY = 0.1

def load_img(name):
    lena = mpimg.imread(name)
    return lena
    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def place_obs(im, obs,r):
    [x_size, y_size] = np.shape(im)
    ID = [int(np.round(obs[0]/resX)),int(np.round(obs[1]/resY))]
    for i in range(r):
        for j in range(r-i):
            if (not ID[0]-i < 0) and (not ID[1]-j < 0):                
                im[ID[0]-i][ID[1]-j] = 0
            if (not ID[0]+i >= x_size) and (not ID[1]-j < 0):     
                im[ID[0]+i][ID[1]-j] = 0
            if (not ID[0]-i < 0) and (not ID[1]+j >= y_size):    
                im[ID[0]-i][ID[1]+j] = 0
            if (not ID[0]+i >= x_size) and (not ID[1]+j >= y_size):
                im[ID[0]+i][ID[1]+j] = 0    
    return im


if __name__ == '__main__':
    image = load_img(img_name)
    image_g = rgb2gray(image)
    [x_size, y_size] = np.shape(image_g)
    im_c = [[1 for i in range(x_size)] for j in range(y_size)]
    for i in range(0,x_size):
        for j in range(0,y_size):
            im_c[j][x_size-i-1] = round(image_g[i,j])
    im_c = place_obs(im_c,[10,10],10)
    im_p = [[1 for i in range(x_size)] for j in range(y_size)]
    for i in range(0,x_size):
        for j in range(0,y_size):
            im_p[j][i] = im_c[i][j]
    plt.imshow(im_p, cmap='gray', origin='lower')
    fig_range = [-20,120,-20,120]
    plt.axis(fig_range)
    plt.show()

