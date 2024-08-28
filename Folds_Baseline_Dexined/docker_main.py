"""
SegcCol challenge - MICCAI 2024
Challenge link: 

This is a dummy example to illustrate how participants should format their prediction outputs.
"""
import numpy as np
import glob
from PIL import Image
from torchvision import transforms
import os
import sys

from Folds_Baseline_Dexined.model import DexiNed
import torch
import cv2
from Folds_Baseline_Dexined.utils.image import image_normalization



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


def predict_masks(pred_file, output_file, model, device):
    """
    param im_path: Input path for a single image
    param output_folder: Path to folder where output will be saved
    predict the segmentation masks for an image and save in the correct formatting as .npy file
    """
    ### replace below with your own prediction pipeline ###
    
    # set default parameters
    mean_pixel_values =[103.939,116.779,123.68, 137.86]
    mean_bgr = mean_pixel_values[0:3]

    # read and process image
    img = cv2.imread(pred_file)
    img = np.array(img, dtype=np.float32)
    img -= mean_bgr
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy()).float().to(device).unsqueeze(0)
    
    # predict mask
    pred_orig = model(img)
    
    # Apply torch.sigmoid to each tensor in pred_orig
    sigmoid_applied = [torch.sigmoid(tensor) for tensor in pred_orig]
    # Stack the tensors and calculate the mean along the desired dimension
    pred_mean = torch.mean(torch.stack(sigmoid_applied), dim=0)

    ## Since only dexined is used, we only have fold prediction, and the other classes channels will be set to zero for this experiment
    prediction_fold = pred_mean.cpu().detach().numpy().astype(np.float64)
    prediction_fold_trans = prediction_fold.squeeze(0).transpose(1,2,0)
    # Create an array of zeros with the same height and width as prediction_fold_trans but with 3 additional channels
    zeros = np.zeros((*prediction_fold_trans.shape[:2], 3))
    # Concatenate the original data with the zeros along the third dimension
    prediction_4ch = np.concatenate((prediction_fold_trans, zeros), axis=-1)
    # normalize images and convert to float64
    prediction = image_normalization(prediction_4ch, 0, 1).astype(np.float64)
    
    ### End of prediction pipeline ###

    # Save the predictions in the correct format
    # Change the format of prediction to nparray and save
    assert type(prediction) == np.ndarray, \
        "Wrong type of predicted depth, expected np.ndarray, got {}".format(type(prediction))
    
    # asset prediction is between 0 and 1
    assert np.max(prediction) <= 1 and np.min(prediction) >= 0, \
        "Wrong range of predicted depth, expected [0, 1], got [{}, {}]".format(np.min(prediction), np.max(prediction))
    
    assert prediction.shape == (480, 640, 4), \
        "Wrong size of predicted depth, expected (480, 640, 4), got {}".format(list(prediction.shape))
    
    assert prediction.dtype == np.float64, \
        "Wrong data type of predicted depth, expected np.float64, got {}".format(prediction.dtype)
 
    np.save(output_file, prediction)
    print(f"Input prediction file: {pred_file}")
    print(f"Saved prediction to: {output_file}")


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(input_path):
        print('No input folder found')
        exit(1)
    else:
        print(input_path +' exists')

    # Added to load the model
    checkpoint_path = "Folds_Baseline_Dexined/checkpoints/SEGCOL/16/16_model.pth"
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    model = DexiNed().to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte not found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))
    model.eval()
    # End of model loading

    # iterate over all images in the input folder
    for seq_path in np.sort(glob.glob(f"{input_path}/Seq*")):
        input_file_list = np.sort(glob.glob(f"{seq_path}/imgs/*.png"))
        for i, input_file in enumerate(input_file_list):
            input_image_name = os.path.basename(input_file)
            # Extract the base part of the filename
            base_name = input_image_name.split('_')[0]
            # Generate the new filename with a sequential number
            output_image_name_number = f'{i}.npy'
            output_image_name = f'Seq{input_file.split("Seq")[1].replace("/", "_").replace("imgs", "segm_map").split("frame")[0]}{output_image_name_number}'
            output_file = input_file.replace(input_path, output_path).replace('imgs', 'predictions').replace(input_image_name, output_image_name)
            output_folder = os.path.dirname(output_file)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(output_folder +' created')
            # Predict and save mask
            predict_masks(input_file_list[i], output_file, model, device)