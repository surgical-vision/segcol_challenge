import os

import cv2
import numpy as np
import torch
import kornia as kn


def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img

def count_parameters(model=None):
    if model is not None:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        print("Error counting model parameters line 32 img_processing.py")
        raise NotImplementedError

def convert_maps_to_rgbs(binary_maps):
    # Step 2: Convert the binary maps to RGB images
    # Define the colors
    colors = {
        'folds': (255, 255, 255),
        'balloon': (127, 127, 127),
        'captivator': (128, 128, 128),
        'forceps': (129, 129, 129)
    }



    # Initialize the RGB image for the current item
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Assign colors based on the binary maps
    rgb_image[binary_maps[0] == 1] = colors['folds']
    rgb_image[binary_maps[1] == 1] = colors['balloon']
    rgb_image[binary_maps[2] == 1] = colors['captivator']
    rgb_image[binary_maps[3] == 1] = colors['forceps']
    

    
    return rgb_image

def save_image_batch_to_disk(tensor, output_dir, file_names, img_shape=None, arg=None, is_inchannel=False):

    os.makedirs(output_dir, exist_ok=True)

    # 255.0 * (1.0 - em_a)
    edge_maps = []
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)
 
    image_shape = tensor.shape[0] * [[tensor.shape[3], tensor.shape[2]]]
    assert len(image_shape) == len(file_names)

    idx = 0
    for file_name in file_names:
        tmp = tensor[idx, :, ...]
        tmp = np.squeeze(tmp)
        
        tmp_img = tmp
        tmp_img = np.uint8(image_normalization(tmp_img))
        tmp_img = cv2.bitwise_not(tmp_img)
        pred = tmp_img

        pred_np = np.array(pred, dtype=np.uint8)
        mask = pred_np/255
        mask =  (mask > 0.9).astype(np.uint8)
        final_img = convert_maps_to_rgbs(mask)
        output_file_name = os.path.join(output_dir, "/".join(file_name.split("/")[4:]))
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

        cv2.imwrite(output_file_name, final_img)
        np.save(os.path.splitext(output_file_name)[0] + ".npy", 1-(pred_np/255))

        idx += 1




def restore_rgb(config, I, restore_rgb=False):
    """
    :param config: [args.channel_swap, args.mean_pixel_value]
    :param I: and image or a set of images
    :return: an image or a set of images restored
    """

    if len(I) > 3 and not type(I) == np.ndarray:
        I = np.array(I)
        I = I[:, :, :, 0:3]
        n = I.shape[0]
        for i in range(n):
            x = I[i, ...]
            x = np.array(x, dtype=np.float32)
            x += config[1]
            if restore_rgb:
                x = x[:, :, config[0]]
            x = image_normalization(x)
            I[i, :, :, :] = x
    elif len(I.shape) == 3 and I.shape[-1] == 3:
        I = np.array(I, dtype=np.float32)
        I += config[1]
        if restore_rgb:
            I = I[:, :, config[0]]
        I = image_normalization(I)
    else:
        print("Sorry the input data size is out of our configuration")
    return I


def visualize_result(imgs_list, arg):
    """
    data 2 image in one matrix
    :param imgs_list: a list of prediction, gt and input data
    :param arg:
    :return: one image with the whole of imgs_list data
    """
    n_imgs = len(imgs_list)
    data_list = []
    for i in range(n_imgs):
        tmp = imgs_list[i]
        # print(tmp.shape)
        if tmp.shape[0] == 3:
            tmp = np.transpose(tmp, [1, 2, 0])
            tmp = restore_rgb([
                arg.channel_swap,
                arg.mean_pixel_values[:3]
            ], tmp)
            tmp = np.uint8(image_normalization(tmp))
        else:
            tmp = np.squeeze(tmp)
            if len(tmp.shape) == 2:
                tmp = np.uint8(image_normalization(tmp))
                tmp = cv2.bitwise_not(tmp)
                tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
            else:
                tmp = np.uint8(image_normalization(tmp))
        data_list.append(tmp)
        # print(i,tmp.shape)
    img = data_list[0]
    if n_imgs % 2 == 0:
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1]
                         * (n_imgs // 2) + ((n_imgs // 2 - 1) * 5), 3))
    else:
        imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1]
                         * ((1 + n_imgs) // 2) + ((n_imgs // 2) * 5), 3))
        n_imgs += 1

    k = 0
    imgs = np.uint8(imgs)
    i_step = img.shape[0] + 10
    j_step = img.shape[1] + 5
    for i in range(2):
        for j in range(n_imgs // 2):
            if k < len(data_list):
                imgs[i * i_step:i * i_step+img.shape[0],
                     j * j_step:j * j_step+img.shape[1],
                     :] = data_list[k]
                k += 1
            else:
                pass
    return imgs