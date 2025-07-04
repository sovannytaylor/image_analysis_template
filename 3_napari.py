"""Quality control: use napari to validate cellpose-generated masks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import napari
from loguru import logger
from skimage.segmentation import clear_border
from napari.settings import get_settings
get_settings().application.ipy_interactive = False
logger.info('Import ok')

image_folder = 'results/initial_cleanup/'
mask_folder = 'results/cellpose_masking/'
output_folder = 'results/napari_masking/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

def filter_masks(image, image_name, mask):
    """Quality control of cellpose-generated masks
    - Select the cell layer and using the fill tool set to 0, remove all unwanted cells.
    - Finally, using the brush tool add or adjust any masks within the appropriate layer.

    Args:
        before_image (ndarray): self explanatory
        image_name (str): self explanatory
        mask (ndarray): self explanatory

    Returns:
        ndarray: stacked masks
    """
    #np.zeros default is float - needed to change to integer 
    cells = mask[0, :, :].copy()
    nuclei = mask[1, :, :].copy()
    
    # create the viewer and add the image
    viewer = napari.Viewer()
    viewer = napari.view_image(image, name='raw_image')
    
    # add the labels
    viewer.add_labels(cells, name='cells')
    viewer.add_labels(nuclei, name='nuclei')

    napari.run()

    np.save(f'{output_folder}{image_name}_mask.npy',
            np.stack([cells, nuclei]))
    logger.info(
        f'Processed {image_name}. Mask saved to {output_folder}{image_name}')

    return np.stack([cells, nuclei])


def stack_channels(name, masks_filtered, cells_filtered_stack):
    masks_filtered[name] = cells_filtered_stack

# ----------------Initialise file list----------------
# read in numpy masks
nuc_masks = np.load(f'{mask_folder}cellpose_nucmasks.npy')
cell_masks = np.load(f'{mask_folder}cellpose_cellmasks.npy')

# clean filenames
file_list = [filename for filename in os.listdir(
    image_folder) if 'npy' in filename]

# 0 = before; 1 = after; 2 = h342; 3 = POL
images = {filename.replace('.npy', ''): np.load(
    f'{image_folder}{filename}') for filename in file_list}

mask_stacks = {
    image_name: np.stack([cell_masks[x], nuc_masks[x]])
    for x, image_name in (enumerate(images.keys()))}

# ---------------- filtering out border cells ----------------
# make new dictionary to check for saturation
image_names = images.keys()
image_values = zip(images.values(), mask_stacks.values())
remove_border_cells = dict(zip(image_names, image_values))

logger.info('removing border cells')
masks_filtered = {}
for name, image in remove_border_cells.items():
    # remove cells near border
    cells_filtered = clear_border(image[1][0, :, :], buffer_size=10)

    # keep objects larger than minimum size
    cells_filtered = ski.morphology.remove_small_objects(cells_filtered, min_size=1000)

    # keep intracellular nuclei
    intra_nuclei = np.where(cells_filtered >= 1, image[1][1, :, :], 0)
    
    # stack the filtered masks
    cells_filtered_stack = np.stack((cells_filtered.copy(), intra_nuclei.copy()))
    stack_channels(name, masks_filtered, cells_filtered_stack)

# ----------------Manually filter masks----------------
already_filtered_masks = [filename.replace('_mask.npy', '') for filename in os.listdir(
    f'{output_folder}') if '_mask.npy' in filename]
unval_images = dict([(key, val) for key, val in images.items() if key not in already_filtered_masks])

filtered_masks = {}
for image_name, image_stack in unval_images.items():
    image_name
    mask_stack = masks_filtered[image_name].copy()
    filtered_masks[image_name] = filter_masks(
        image_stack, image_name, mask_stack)

#### ----------------------- EXPLANATION OF HOW NAPARI GUI WORKS -----------------------
# Once you run the above chunk, a napari window will open and everytime you close it, the next image will appear and the mask you just edited/ looked at will save 

#### How to stop but not save - hit interrupt in the interactive window to close napari without saving the current image you are at 

##### Each mask number is unique to each image so labels just need to be unique per image not per entire set of images 

#### Any changes to the image in terms of brightness/ contrast is not saved to the raw data - the contrast changes are only for visualization, but do not affect downstream quantifications 