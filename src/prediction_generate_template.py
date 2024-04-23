"""
SegcCol challenge - MICCAI 2024
Challenge link: 
Task 1: Segmentation in simulated colonoscopy
Task 2: 

This is a dummy example to illustrate how participants should format their prediction outputs.
"""
import numpy as np
import glob
from PIL import Image
from torchvision import transforms
import os
import sys
from skimage.util import random_noise


to_tensor = transforms.ToTensor()

def get_images(file):
    # Open the image
    img = Image.open(file)

    # Convert the image data to a numpy array
    img_data = np.array(img)


    # Create class channels using np.where
    class1 = np.where(img_data == 255, 1, 0) # fold
    class2 = np.where(img_data == 127, 1, 0) # tool1 
    class3 = np.where(img_data == 128, 1, 0) # tool2
    class4 = np.where(img_data == 129, 1, 0) # tool3

    # Stack all class channels together
    img_data = np.stack((class1, class2, class3, class4), axis=-1)

    return img_data

def predict_masks(pred_file, output_file):
    """
    param im_path: Input path for a single image
    param output_folder: Path to folder where output will be saved
    predict the segmentation masks for an image and save in the correct formatting as .npy file
    """
    ### replace below with your own prediction pipeline ###
    # We generate dummy predictions here with noise added to ground truth
    
    pred_orig = get_images(pred_file).astype(np.float16) #dummy make sure pred is float

    prediction = random_noise(pred_orig, mode='gaussian', var=0.5, rng = 42)

    ### End of prediction pipeline ###

    # Save the predictions in the correct format
    # Change the format of predction to nparray and save
    assert prediction.shape == (480, 640, 4), \
        "Wrong size of predicted depth, expected (480, 640, 4), got {}".format(list(prediction.shape))
    
 
    np.save(output_file, prediction)
    print(f"Saved prediction to {output_file}")


if __name__ == "__main__":
    main_path = sys.argv[1]

    if not os.path.exists(main_path):
        print('No input folder found')
    else:
        print(main_path +' exists')
        glob.__file__
        input_file_list = np.sort(glob.glob(f"{main_path}/Seq*/segm_maps/*.png"))


    for i in range(len(input_file_list)):
        output_file = input_file_list[i].replace('segm_maps','predictions').replace('.png','.npy')
        output_folder = os.path.dirname(output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(output_folder +' created')
        predict_masks(input_file_list[i], output_file)
