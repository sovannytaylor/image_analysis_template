"""Applies cellpose algorithms to determine cellular and nuclear masks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose import plot
from loguru import logger
from scipy import ndimage as ndi
from skimage import (
    exposure,filters, measure, morphology, segmentation
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk, ball
from skimage.filters.rank import equalize
from skimage.filters import rank
from cellpose.io import logger_setup
logger_setup();

logger.info('Import ok')

input_folder = 'python_results/initial_cleanup/'
output_folder = 'python_results/cellpose_masking/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def apply_cellpose(images, image_type='cyto', channels = None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, resample=False):
    """Apply standard cellpose model to list of images.

    Args:
        images (ndarray): numpy array of 16 bit images
        image_type (str, optional): Cellpose model. Defaults to 'cyto'.
        channels (int, optional): define CHANNELS to run segementation on (grayscale=0, R=1, G=2, B=3) where channels = [cytoplasm, nucleus]. Defaults to None.
        diameter (int, optional): Expected diameter of cell or nucleus. Defaults to None.
        flow_threshold (float, optional): maximum allowed error of the flows for each mask. Defaults to 0.4.
        cellprob_threshold (float, optional): The network predicts 3 outputs: flows in X, flows in Y, and cell “probability”. The predictions the network makes of the probability are the inputs to a sigmoid centered at zero (1 / (1 + e^-x)), so they vary from around -6 to +6. Decrease this threshold if cellpose is not returning as many ROIs as you expect. Defaults to 0.0.
        resample (bool, optional): Resampling can create smoother ROIs but take more time. Defaults to False.

    Returns:
        ndarray: array of masks, flows, styles, and diameters
    """
    if channels is None:
        channels = [0, 0]
    model = models.Cellpose(model_type=image_type)
    masks, flows, styles, diams = model.eval(
        images, diameter=diameter, channels=channels, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, resample=resample)
    return masks, flows, styles, diams


def visualise_cell_pose(images, masks, flows, channels = None):
    """Display cellpose results for each image

    Args:
        images (ndarray): single channel (one array)
        masks (ndarray): one array
        flows (_type_): _description_
        channels (_type_, optional): _description_. Defaults to None.
    """
    if channels is None:
        channels = [0, 0]
    for image_number, image in enumerate(images):
        maski = masks[image_number]
        flowi = flows[image_number][0]
        
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, image, maski, flowi, channels=channels)
        plt.tight_layout()
        plt.show()


# ----------------Initialise file list----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

# reading in both channels for each image
imgs = [np.load(f'{input_folder}{filename}')
        for filename in file_list]

# need a library?
images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

# ----------------Outline Cells-----------------
ch2_channel = [image[1, :, :] for name, image in images.items()]

# segment cytoplasm
masks, flows, styles, diams = apply_cellpose(
        ch2_channel, image_type='cyto', diameter=200, flow_threshold=0.0, resample=True, cellprob_threshold=0)
visualise_cell_pose(ch2_channel, masks, flows, channels=[0, 0]) 

#**dont forget to add it to ch2_channel on this function statement 
## when trying to test, put [:4] or however many at the input so after apply cell pose 
##after running ch2, do plt.imshow for the first image in the list and determine cell diameter 

# save cell masks before moving on to nuclei (for memory)
np.save(f'{output_folder}cellpose_cellmasks.npy', masks)
logger.info('cell masks saved')


# ----------------Outline nuclei----------------
ch2 = [image[1, :, :] for name, image in images.items()]


## when trying to test, put [:4] or however many at the input so after apply cell pose 
##after running ch4, do plt.imshow for the first image in the list and determine cell diameter 

##visualise_cell_pose(ch4[:4], nuc_masks, flows, channels=[0, 0]) 

# to check the diameter - do plt.imshow(images['name'[channel]])

# segment nuclei
nuc_masks, flows, styles, diams = apply_cellpose(
        ch2, image_type='nuclei', diameter=125, flow_threshold=0.4, resample=True, cellprob_threshold=0)
visualise_cell_pose(ch2, nuc_masks, flows, channels=[0, 0])

# save nuclei masks
np.save(f'{output_folder}cellpose_nucmasks.npy', nuc_masks)
logger.info('nuclei masks saved')