"""
SegcCol challenge - MICCAI 2024
Challenge link:
 
This is a dummy example to illustrate how participants should format their prediction outputs.
"""
import numpy as np
import glob
import os
import sys
import random
 
 
def predict_masks(img_files, output_path):
    """
    param im_path: Input path for all images
    param output_folder: Path to folder where output will be saved
    select the best subset from the input images save in a text file samples.txt
    """
    ### replace below with your own prediction pipeline ###
    # We generate random selections here
   
    # Set the seed for reproducibility
    random.seed(42)
 
    # Calculate the number of elements to select
    percentage = random.uniform(0.5, 0.9)  # Random percentage between 50% and 90%
    num_elements = int(percentage * len(img_files))
    
    # Select the elements
    selected_imgs = random.sample(img_files, num_elements)
    
    ### End of prediction pipeline ###
 
    # Save the selected images to a text file
    with open(os.path.join(output_path, 'samples_xtest.txt'), 'w') as f:
        f.writelines(f"Seq{img.split('Seq')[1]}\n" for img in selected_imgs)

    # This will save your output file within the docker image for submission
    with open('/opt/samples_xtest.txt', 'w') as f:
        f.writelines(f"{img}\n" for img in selected_imgs)
   
    print(f"Saved selections to {os.path.join(output_path, 'samples_xtest.txt')}")
    print(f"Saved selections to /opt/samples_xtest.txt")
 
 
if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
 
    if not os.path.exists(input_path):
        print('No input folder found')
    else:
        print(input_path +' exists')
        input_file_list = sorted(glob.glob(f"{input_path}/Seq*/imgs/*.png"))
       
       
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(output_path +' created')
           
    predict_masks(input_file_list, output_path)