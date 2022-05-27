import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from skimage import data

from skimage import io
import cv2
import commonfunctions as cf

image = io.imread("M14.jpg")
image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cf.show_images([image], ['GLCM image'])

def GLCM (image):

    #<matplotlib.image.AxesImage at 0xf695f98>
    PATCH_SIZE = 21

    # open the camera image
    #image = data.camera()

    # select some patches from grassy areas of the image
    grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
    grass_patches = []
    for loc in grass_l 
    for patch in (grass_patches + sky_patches):
        glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
        xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(graycoprops(glcm, 'correlation')[0, 0])

    # create the figure
    fig = plt.figure(figsize=(8, 8))

    # display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(image, cmap=plt.cm.gray,
            vmin=0, vmax=255)
    for (y, x) in grass_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
    for (y, x) in sky_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
            label='Grass')
    ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
            label='Sky')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLCM Correlation')
    ax.legend()

    # display the image patches
    for i, patch in enumerate(grass_patches):
        ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                vmin=0, vmax=255)
        ax.set_xlabel('Grass %d' % (i + 1))

    for i, patch in enumerate(sky_patches):
        ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                vmin=0, vmax=255)
        ax.set_xlabel('Sky %d' % (i + 1))


    # display the patches and plot
    fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
GLCM(image)    