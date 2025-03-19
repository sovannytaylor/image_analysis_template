import os
import numpy as np
from loguru import logger
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter

logger.info('Import ok')

input_path = 'raw_data'
output_folder = 'python_results/initial_cleanup/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

def czi_converter(image_name, input_folder, output_folder, tiff=False, array=True):
    """Stack images from nested .czi files and save for subsequent processing

    Args:
        image_name (str): image name (usually iterated from list)
        input_folder (str): filepath
        output_folder (str): filepath
        tiff (bool, optional): Save tiff. Defaults to False.
        array (bool, optional): Save np array. Defaults to True.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # import image
    image = AICSImage(
            f'{input_folder}/{image_name}.czi').get_image_data("CYX", B=0, Z=0, V=0, T=0)

    if tiff == True:
        # Save image to TIFF file with image number
        OmeTiffWriter.save(
            image, f'{output_folder}{image_name}.tif', dim_order='CYX')

    if array == True:
        np.save(f'{output_folder}{image_name}.npy', image)


# ---------------Initalize file_list---------------
# read in all folders with image files (LSM900)
file_list = [[f'{filename}' for filename in files if '.czi' in filename]
    for root, dirs, files in os.walk(f'{input_path}')]

# flatten file_list
file_list = [item for sublist in file_list for item in sublist]

do_not_quantitate = ['555']

# ---------------Collect image names & convert---------------
image_names = []
for filename in file_list:
    if all(word not in filename for word in do_not_quantitate):
        filename = filename.split('.czi')[0]
        image_names.append(filename)

# remove duplicates
image_names = list(dict.fromkeys(image_names))

# collect only before and after images
for name in image_names:
    name
    czi_converter(name, input_folder=f'{input_path}',
                    output_folder=f'{output_folder}')

logger.info('initial cleanup complete :)')
