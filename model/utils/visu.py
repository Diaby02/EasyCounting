import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('agg')
from torch import nn
from torchinfo import summary
import os
import torch
from PIL import Image
from utils.utils_ import createFolder
from pathlib import Path

IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

#--------------------------------------#

# Visualize the image with the bounding boxes

#--------------------------------------#

def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):
    """Reverses the normalisation on a tensor.
    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.
    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean
    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor
    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)
    Return:
        torch.Tensor (torch.float32): Demornalised image tensor with pixel
            values between [0, 1]
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)

    return denormalized


def scale_and_clip(val, scale_factor, min_val, max_val):
    "Helper function to scale a value and clip it within range"

    new_val = int(round(val*scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val

def full_vizu(input_image, output, boxes, init_boxes, rslt_path, model_name,pred_cnt=None, peaks=None, gt=None):
    """
    Visualize the output of the model

    Args:
        input_image (str): path to the input image
        output (torch.Tensor): output of the model, of shape (1, 1, H, W) (apply torch.detach() before passing it to this function + convert to cpu)
        boxes (torch.Tensor): bounding boxes of the objects, of shape (1, N, 4) (apply torch.detach() before passing it to this function + convert to cpu)
        init_boxes (torch.Tensor): initial bounding boxes of the objects, of shape (N, 4)
        rslt_path (str): path to save the results
        image_name (str): name of the image
        model_name (str): name of the model
    
    """
    nb_map = 4
    image_name = Path(input_image).stem
    rslts_files = [os.path.join(rslt_path, image_name + "_map_" + str(i) + ".png") for i in range(nb_map)]

    #first map
    simple_vizu(input_image, output, rslts_files[0],pred_cnt=pred_cnt, peaks=peaks)

    # second map
    simple_vizu(input_image, output, rslts_files[1], title=model_name,pred_cnt=pred_cnt, peaks=peaks)

    # third map
    visualize_with_img_and_boxes(input_image, output, boxes, init_boxes, rslts_files[2],pred_cnt=pred_cnt, peaks=peaks, gt=gt)

    # fourth map
    simple_vizu_with_box(input_image, output, boxes, init_boxes, rslts_files[3],pred_cnt=pred_cnt)

    return

def partial_vizu(input_image, output, boxes, init_boxes, rslt_path,model_name,pred_cnt=None, peaks=None, gt=None):

    nb_map = 2
    image_name = Path(input_image).stem
    rslts_files = [os.path.join(rslt_path, image_name + "_map_" + str(i) + ".png") for i in range(nb_map)]

    # first map
    simple_vizu(input_image, output, rslts_files[0], title=model_name,pred_cnt=pred_cnt, peaks=peaks)

    display_input(input_image, init_boxes, rslts_files[0], gt=gt)

    # second map
    visualize_with_img_and_boxes(input_image, output, boxes, init_boxes, rslts_files[1],pred_cnt=pred_cnt, peaks=peaks, gt=gt)

    return


def display_input(input_image, init_boxes,rslt_file,gt=0):
    
    img = Image.open(input_image).convert("RGB")
    plt.imshow(img)
    plt.axis('off')
    
    for bbox in init_boxes:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--', facecolor='none')
        plt.gca().add_patch(rect)
        plt.gca().add_patch(rect2)

    if gt is not None:
        plt.title("Input image, GT: "+ str(gt))
    else:
        plt.title("Input image")
    plt.savefig(os.path.join(Path(rslt_file).parent, Path(rslt_file).stem + "_input.png"), bbox_inches="tight",pad_inches=0)
    plt.close()




def visualize_with_img_and_boxes(input_image, output, boxes, init_boxes, rslt_file,pred_cnt=None, figsize=(10, 5), peaks=None, gt=None):
    """
        dots: Nx2 numpy array for the ground truth locations of the dot annotation
            if dots is None, this information is not available
    """

    img = Image.open(input_image).convert("RGB")

    cmap = plt.get_cmap('jet')
    w, h = img.size
    _,_,h_dt,w_dt = output.shape
    origin_img_array = np.array(img)
    # get the total count
    pred_cnt = round(output.sum().item()) if pred_cnt is None else pred_cnt
    boxes = boxes.squeeze(0)

    ########################################################
    # Get the count of each ROI + average size of the object
    ########################################################

    h_avg, w_avg = [], []
    boxes2 = []
    for i in range(0, boxes.shape[0]):
        x1, y1, x2, y2 = int(boxes[i, 0].item()), int(boxes[i, 1].item()), int(boxes[i, 2].item()), int(
            boxes[i, 3].item())
        roi_cnt = output[0,0,y1:y2, x1:x2].sum().item()

        h_avg.append(x2-x1)
        w_avg.append(y2-y1)

        x1, y1, x2, y2 = int(init_boxes[i, 0].item()), int(init_boxes[i, 1].item()), int(init_boxes[i, 2].item()), int(
            init_boxes[i, 3].item())
        boxes2.append((x1,y1,x2,y2,roi_cnt))

    h_avg = np.average(h_avg)
    w_avg = np.average(w_avg)
    fig = plt.figure(figsize=figsize)

    #################################################
    # display the input image with the bounding boxes
    #################################################

    ax = fig.add_subplot(1, 2, 1)
    ax.set_axis_off()
    ax.imshow(img)

    for bbox in init_boxes:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)

    # set the title
    if gt is not None:
        ax.set_title("Input image, GT: "+ str(gt))
    else:
        ax.set_title("Input image")

    #################################################
    # display the density map with the bounding boxes
    #################################################

    ax = fig.add_subplot(1, 2, 2)
    ax.set_axis_off()
    if peaks is not None:
        ax.set_title("Pred_cnt: {:.2f} Pred_cnt_p {:.2f}".format(pred_cnt, len(peaks)))
    else:
        ax.set_title("Pred_cnt: {:.2f}".format(pred_cnt))

    density_map = output
    density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy() #convert the density map to a specific height, squeeze(0) removes the first two dimensions of 1 size of the tensor, places on cpu and converts to numpy array
    density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0 #normalize the density map, add a small constant to avoid / by zero, color the density map and multiply by 255 to get the range of 0-255 (RGB)
    density_map_without_im = density_map[:,:,0:3]
    density_map = density_map[:,:,0:3] * 0.7 + origin_img_array * 0.3 #combine the density map by taking the first 3 channels (like RGB) and multiply by 0.5 and the original image by 0.5

    if peaks is not None:
        scaling_factor_h = h/h_dt
        scaling_factor_w = w/w_dt
        peaks_stretched = [(x*scaling_factor_w, y*scaling_factor_h) for y,x in peaks]
    

    ax.imshow(density_map.astype(np.uint8))
    for bbox in boxes2:
        x1, y1, x2, y2, roi_cnt = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.text(x1, y1, '{:.2f}'.format(roi_cnt), backgroundcolor='y')

    if peaks is not None:
        plt.scatter([x for x,_ in peaks_stretched], [y for _,y in peaks_stretched], c='green', s=30, marker='x')
       
    fig.savefig(rslt_file, bbox_inches="tight")
    #print("===> Image and boxes is saved to {}".format(rslt_file))
    plt.close()

    return

def visualize_output_simple(input_, output, gt, boxes, gt_cnt, save_name, save_path, pred_cnt=None, figsize=(15, 5), dots=None):

    """
        dots: Nx2 numpy array for the ground truth locations of the dot annotation
            if dots is None, this information is not available
    """

    save_path = os.path.join(save_path, "predicted_gt/")
    createFolder(save_path)
    # get the total count
    pred_cnt = round(output.sum().item()) if pred_cnt is None else pred_cnt
    boxes = boxes.squeeze(0)

    boxes2 = []
    for i in range(0, boxes.shape[0]):
        x1, y1, x2, y2 = int(boxes[i, 0].item()), int(boxes[i, 1].item()), int(boxes[i, 2].item()), int(
            boxes[i, 3].item())
        roi_cnt = output[0,0,y1:y2, x1:x2].sum().item()
        boxes2.append([x1, y1, x2, y2, roi_cnt])

    img1 = format_for_plotting(denormalize(input_))
    gt = format_for_plotting(gt)
    output = format_for_plotting(output)

    fig = plt.figure(figsize=figsize)

    # display the input image
    ax = fig.add_subplot(1, 3, 1)
    ax.set_axis_off()
    ax.imshow(img1)

    for bbox in boxes2:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)

    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c='red', edgecolors='blue')
        # ax.scatter(dots[:,0], dots[:,1], c='black', marker='+')
        ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    else:
        ax.set_title("Input image")

    # display the input image
    ax = fig.add_subplot(1, 3, 2)
    ax.set_axis_off()
    ax.set_title("Grount Truth map, count: {:.2f}".format(gt_cnt))
    ax.imshow(gt)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(output)
    for bbox in boxes2:
        x1, y1, x2, y2, roi_cnt = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.text(x1, y1, '{:.2f}'.format(roi_cnt), backgroundcolor='y')

    fig.savefig(save_path+"image_"+save_name, bbox_inches="tight")
    plt.close()

def simple_vizu(input_image, output, rslt_file, title="", pred_cnt=None, peaks=None):

    cmap = plt.get_cmap('jet')
    origin_img = Image.open(input_image).convert("RGB")
    w, h = origin_img.size
    _, _, w_dt, h_dt = output.shape
    origin_img_array = np.array(origin_img)
    pred_cnt = round(output.sum().item()) if pred_cnt is None else pred_cnt
    text = f'{pred_cnt}'

    density_map = output
    density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy() #convert the density map to a specific height, squeeze(0) removes the first two dimensions of 1 size of the tensor, places on cpu and converts to numpy array
    density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0 #normalize the density map, add a small constant to avoid / by zero, color the density map and multiply by 255 to get the range of 0-255 (RGB)
    density_map = density_map[:,:,0:3] * 0.7 + origin_img_array * 0.3 #combine the density map by taking the first 3 channels (like RGB) and multiply by 0.5 and the original image by 0.5

    if peaks is not None:
        scaling_factor_h = h/h_dt
        scaling_factor_w = w/w_dt
        peaks_stretched = [(x*scaling_factor_w, y*scaling_factor_h) for y,x in peaks]
        text = f'{pred_cnt}/{len(peaks)}'

    to_return = density_map.astype(np.uint8)
    plt.imshow(to_return)

    # adapt fontsize and position depending on the image size

    size = h // 18
    size = size // (4/3)

    plt.text(
        x=0.01, y=0.99,  # Adjust x and y to position inside the image
        s=text,
        size=size, color='black', 
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=1),
        ha='left',
        va='top',
        transform=plt.gca().transAxes
    )

    if title != "":
        plt.title(title)

    # Save the image without extra borders
    plt.axis('off')

    if peaks is not None:
        plt.scatter([x for x,_ in peaks_stretched], [y for _,y in peaks_stretched], c='green', s=30, marker='x')
        plt.savefig(os.path.join(Path(rslt_file).parent, Path(rslt_file).stem + "_with_peaks.png"), bbox_inches='tight', pad_inches=0)

    else:
        plt.savefig(os.path.join(Path(rslt_file).parent, Path(rslt_file).stem + ".png"), bbox_inches='tight', pad_inches=0)

    plt.close()

    return

def simple_vizu_with_box(input_image, output, boxes, init_boxes, rslt_file, pred_cnt=None):

    origin_img = Image.open(input_image).convert("RGB")

    cmap = plt.get_cmap('jet')
    w, h = origin_img.size
    origin_img_array = np.array(origin_img)
    # get the total count
    pred_cnt = round(output.sum().item()) if pred_cnt is None else pred_cnt
    boxes = boxes.squeeze(0)

    boxes2 = []
    for i in range(0, boxes.shape[0]):
        x1, y1, x2, y2 = int(boxes[i, 0].item()), int(boxes[i, 1].item()), int(boxes[i, 2].item()), int(
            boxes[i, 3].item())
        roi_cnt = output[0,0,y1:y2, x1:x2].sum().item()

        x1, y1, x2, y2 = int(init_boxes[i, 0].item()), int(init_boxes[i, 1].item()), int(init_boxes[i, 2].item()), int(
            init_boxes[i, 3].item())
        boxes2.append((x1,y1,x2,y2,roi_cnt))

    density_map = output
    density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy() #convert the density map to a specific height, squeeze(0) removes the first two dimensions of 1 size of the tensor, places on cpu and converts to numpy array
    density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0 #normalize the density map, add a small constant to avoid / by zero, color the density map and multiply by 255 to get the range of 0-255 (RGB)
    density_map = density_map[:,:,0:3] * 0.5 + origin_img_array * 0.5 #combine the density map by taking the first 3 channels (like RGB) and multiply by 0.5 and the original image by 0.5

    plt.imshow(density_map.astype(np.uint8))

    for bbox in boxes2:
        x1, y1, x2, y2, roi_cnt = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--',
                                  facecolor='none')
        plt.gca().add_patch(rect)
        plt.gca().add_patch(rect2)
        plt.text(x1, y1, '{:.2f}'.format(roi_cnt), backgroundcolor='y')

    size = h // 14
    size = size // (4/3)

    plt.text(
        x=0.01, y=0.99,  # Adjust x and y to position inside the image
        s=f'{pred_cnt}', 
        size=size, color='black', 
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=1),
        ha='left',
        va='top',
        transform=plt.gca().transAxes
    )

    plt.axis('off')
    plt.savefig(rslt_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("===> Density map with boxes is saved to {}".format(rslt_file))

    return


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.
    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.
    Args:
        tensor (torch.Tensor, torch.float32): Image tensor
    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively
    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()