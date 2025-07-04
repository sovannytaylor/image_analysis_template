import os
import numpy as np
from loguru import logger
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter

logger.info('Import ok')

input_folder = "P:/Sophie/Uptake/Endocytosis/05062025/63x_oil"
output_folder = 'results/initial_cleanup/'
source = input_folder


if not os.path.exists(output_folder):
    os.mkdir(output_folder)

def czi_converter(image_name, input_folder, output_folder, tiff=False, mip=False, array=True):
    """Converts czi files into numpy arrays, if MIP is true will take max intensity projection of each channel and stack"

    Args:
        image_name (str): image
        input_folder (str): path
        output_folder (str): path
        tiff (bool, optional): Return array as tiff. Defaults to False.
        mip (bool, optional): make a maximum intensity projection for each channel. Defaults to False.
        array (bool, optional): _description_. Defaults to True.
    """
 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    image = AICSImage(
        f'{input_folder}/{image_name}.czi').get_image_data("CYX", Z=0, B=0, V=0, T=0)
    
    #make files more human readable 
    short_name = image_name.split('\\')[-1]
      
    if tiff:
        OmeTiffWriter.save(image, f'{output_folder}{short_name}.tif', dim_order='CYX')

    if array:
        np.save(f'{output_folder}{short_name}.npy', image)
    
    if mip:
        image = AICSImage(
        f'{input_folder}/{image_name}.czi').get_image_data("CZYX", B=0, V=0, T=0)

        mip_image = np.max(image, axis=1)
    
        # Save as .npy
        np.save(f'{output_folder}{short_name}_MIP.npy', mip_image)



# ---------------Initialize file_list---------------
if source == 'raw_data':
    flat_file_list = [filename for filename in os.listdir(input_folder) if '.czi' in filename]

else:
    # find directories of interest from shared drive 'M:/Olivia'
    # experiments = ['WT', 'MUT']
    walk_list = [x[0] for x in os.walk(input_folder)]
    walk_list = [item for item in walk_list if any(x in item for x in walk_list)]

    # read in all image files
    file_list = []
    for folder_path in walk_list:
        folder_path
        images = [[f'{root}\{filename}' for filename in files if '.czi' in filename]
            for root, dirs, files in os.walk(f'{folder_path}')]
        file_list.append(images[0])

    # flatten file_list
    flat_file_list = [item for sublist in file_list for item in sublist]

# remove images that do not require analysis (e.g., qualitative controls)
do_not_quantitate = []


# ---------------Collect image names ---------------
image_names = []
for filename in flat_file_list:
    if all(word not in filename for word in do_not_quantitate):
        filename
        filename = filename.split('.czi')[0]
        filename = filename.split(f'{input_folder}')[-1]
        image_names.append(filename)

# remove duplicates
image_names = list(dict.fromkeys(image_names))



# ## Debugging step 

# for name in image_names:
#     img = AICSImage(f'{input_folder}/{name}.czi')
#     try:
#         scales = img.physical_pixel_sizes
#         if None in (scales.X, scales.Y):
#             print(f"Skipping {name} due to missing scaling metadata.")
#             continue
#         image = img.get_image_data("CYX", Z=0, B=0, V=0, T=0)
#     except Exception as e:
#         print(f"Skipping {name} due to unexpected error: {e}")
#         continue

# ---------------Run Conversions---------------

log_file = 'processed_files.txt'

# Load already processed files
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        processed_files = set(line.strip() for line in f)
else:
    processed_files = set()

for name in image_names:
    if name in processed_files:
        print(f"Skipping {name}, already processed.")
        continue

    try:
        czi_converter(name, input_folder=input_folder, output_folder=output_folder, mip=True)
        # Log the successfully processed file
        with open(log_file, 'a') as f:
            f.write(f"{name}\n")
    except Exception as e:
        print(f"Error processing {name}: {e}")


logger.info('initial cleanup and MIP generation complete :)')




