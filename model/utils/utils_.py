from pathlib import Path
import time
import hdbscan
import numpy as np
import pandas as pd
import torch.nn.functional as F
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# detection methods 
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import *
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

matplotlib.use('agg')
from torch import nn
from torchinfo import summary
import os
import torch
import skimage
import scipy
from PIL import Image

IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)

##### Infos about a model

def getInfoModel(model,img,bbox,bboxes):
    
    args = [img,bbox,bboxes]
    dtypes = [torch.float32]
    model_summary = summary(model, dtypes=dtypes, verbose=0,
                            col_names=["input_size","output_size","num_params","kernel_size"], input_data=args)
    print(model_summary)
    return model_summary

# source: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
    return result

#--------------------------------------#

# NMS on output of the model

#--------------------------------------#

def find_peaks(out,boxes, threshold=0.1, mode='average'):
    dt_array = out.squeeze(0).squeeze(0).numpy()
    widths = [int(box[2].item())-int(box[0].item()) for box in boxes.squeeze(0)]
    heights = [int(box[3].item())-int(box[1].item()) for box in boxes.squeeze(0)]
    
    if mode == 'average':
        width = np.average(widths)
        height = np.average(heights)
    else:
        width = min(widths)
        height = min(heights)
        
    radius = int(min(height,width)/2)
    peaks = skimage.feature.peak_local_max(dt_array,min_distance=radius, exclude_border=0, threshold_rel=threshold)
    return peaks, len(peaks), width, height

#--------------------------------------#

# Loss function, MSE normalized by the number of objects

#--------------------------------------#
class ObjectNormalizedL2Loss(nn.Module):

    def __init__(self):
        super(ObjectNormalizedL2Loss,self).__init__()

    def forward(self, output, dmap, num_objects):
        return ((output - dmap) ** 2).sum() / num_objects


#-----------------------------------------------------------#

# Density map comparison (clustering, hungarian matching,...)

#-----------------------------------------------------------#

# BBDR and BBMAE computation
def compute_bbr_bbmape(out,points,resultsPath,im_path,image_size,patch_size):
    """
    Computes the BBDR and BBMRE metrics for the given output and ground truth density map.
    Args:
        out (torch.Tensor): The output density map. size: (H,W)
        points (torch.Tensor): List of points to compute the metrics. size (nb_points,2)
        resultsPath (str): Path to save the results.
        im_path (str): Path of the image.
        image_size (int): Size of the image.
        patch_size (int): Size of the patch.
        visu (bool): Whether to visualize the results or not.
    Returns:
        None: The results are saved in the resultsPath with two csv files:
        - bbdr.csv: Contains the BBDR metrics.
        - bbmre.csv: Contains the BBMRE metrics.
    """
    image_name = str(Path(im_path).stem)
    rslt_path1 = os.path.join(resultsPath, "bbdr")
    rslt_path2 = os.path.join(resultsPath, "bbmape")

    bboxes_dt  = []
    bboxes_mre = []

    for i in range(4,29,4):
        total_density = 0
        total_mre = 0
        nb_points = 0
        for x,y in points:
            if (x,y) == (0,0):
                break
            nb_points +=1
            x1, y1, x2, y2 = int(x-(i/2)),int(y-(i/2)),int(x+(i/2)),int(y+(i/2))
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > image_size:
                x2 = image_size
            if y2 > image_size:
                y2 = image_size

            density_box = out[y1:y2,x1:x2].sum().item()
            mre_box = abs(1 - density_box)
            total_density += density_box
            total_mre += mre_box
        
        total_density = total_density / out.sum().item()
        total_mre = total_mre/nb_points
        bboxes_dt.append(round(total_density,3))
        bboxes_mre.append(round(total_mre,3))

    columns_name = ["image_name","image_size","patch_size", "box_1", "box_2", "box_3", "box_4", "box_5", "box_6", "box_7"]
    bboxes_dt, bboxes_mre = [image_name] + [image_size, patch_size] + bboxes_dt, [image_name] + [image_size, patch_size] + bboxes_mre

    if (os.path.exists(rslt_path1+".csv")):
        df1 = pd.read_csv(rslt_path1+".csv")
        row_to_add = pd.DataFrame([bboxes_dt],columns=columns_name)
        df1  = pd.concat([df1, row_to_add], ignore_index=True)
    else:
        df1 = pd.DataFrame([bboxes_dt],columns=columns_name)

    df1.to_csv(rslt_path1+".csv", index=False)

    if (os.path.exists(rslt_path2+".csv")):
        df2 = pd.read_csv(rslt_path2+".csv")
        row_to_add = pd.DataFrame([bboxes_mre],columns=columns_name)
        df2  = pd.concat([df2, row_to_add], ignore_index=True)
    else:
        df2 = pd.DataFrame([bboxes_mre],columns=columns_name)

    df2.to_csv(rslt_path2+".csv", index=False)

#####################################################
#       Optimal Transport Minimization (OTM)        #
#####################################################

def compute_otm(out,gt_points,visu=False):
    """
    Computes the Optimal Transport Minimization (OTM) for the given output and ground truth points.
    Args:
        out (torch.Tensor): The output density map. size: (H,W)
        gt_points (np.ndarray): Ground truth points. size: (nb_points,2)
        visu (bool): Whether to visualize the results or not.
    Returns:
        predicted_points (np.ndarray): The predicted points after applying the OTM. size: (nb_predicted_points,2)
        len(predicted_points) (int): The number of predicted points.
        total_time (float): The total time taken to compute the OTM.
    """
    start = time.time()
    # transform the density map into a list of points
    points,_,_ = den2points(out)
    predicted_points = den2seq(out, scale_factor=1, max_itern=16, ot_scaling=0.75)

    # exchange x and y
    predicted_points = predicted_points.tolist()
    predicted_points = [coord[::-1] for coord in predicted_points]
    predicted_points = np.array(predicted_points)

    end = time.time()
    total_time = round(end-start,3)
    print("Total time to compute Optimal Transport: ",str(total_time)," s")

    if visu:
        plt.figure(figsize=(6, 6))
        plt.gca().invert_yaxis()
        plt.scatter(points[:, 1], points[:, 0])
        plt.scatter(predicted_points[:, 0], predicted_points[:, 1], s=20, c=["orange"]*len(predicted_points)) # attention: inverted x and y for predicted points, same as with peaks local max
        plt.scatter(gt_points[:, 0], gt_points[:, 1], s=20, c=["red"]*len(gt_points))
        plt.title("Predicted_blobs")
        plt.tight_layout()
        plt.savefig("test.png")
        plt.close()

    return predicted_points, len(predicted_points), total_time

##################################################################################################
#       Hierachical Density-based Spatial Clustering of Applications with Noise (HDBSCAN)        #
##################################################################################################

def compute_hdbscan(out,im_path,resultsPath,min_object_size,gt_points,visu=False):
    """
    Computes the HDBSCAN clustering algorithm for the given output and ground truth points.
    Args:
        out (torch.Tensor): The output density map. size: (H,W)
        im_path (str): Path of the image, used to get the name of the image.
        resultsPath (str): Directory path to save the results.
        min_object_size (int): Minimum size of the object to be detected.
        gt_points (np.ndarray): Ground truth points. size: (nb_points,2)
        visu (bool): Whether to visualize the results or not.
    Returns:
        centers (np.ndarray): The centers of the clusters detected by HDBSCAN. size: (nb_clusters,2)
        len(centers) (int): The number of clusters detected.
        total_time (float): The total time taken to compute the HDBSCAN.
    """
    image_name = str(Path(im_path).stem)
    rslt_path1 = os.path.join(resultsPath, image_name + "_hdbscan")

    start = time.time()
    points, _, points_bool = den2points(out)

    #weighting
    total_points = points.tolist()
    max_value = round(torch.max(out).item(),4)
    range_of_value = [round(max_value/10,4)*i for i in range(10)]

    for h in range(out.shape[0]):
        for w in range(out.shape[1]):
            if points_bool[h][w] == 1:
                for ind, value in enumerate(range_of_value):
                    if ind != len(range_of_value)-1:
                        if out[h][w] > value and out[h][w] < range_of_value[ind+1] and ind != 0:
                            total_points = total_points + [[h,w]]*ind
                    else:
                        if out[h][w] > value:
                            total_points = total_points + [[h,w]]*ind

    total_points = np.array(total_points)
    model = hdbscan.HDBSCAN(cluster_selection_epsilon=min_object_size/2,min_cluster_size=round(min_object_size*min_object_size/4),cluster_selection_epsilon_max=min_object_size).fit(total_points)

    centers = np.empty(shape=(max(model.labels_)+1, 2))
    for i in range(max(model.labels_)+1):
        centers[i, :] = model.weighted_cluster_centroid(i)
    end = time.time()
    total_time = round(end-start,3)
    print("Total time to compute HDBSCAN: ",str(total_time)," s")

    # exchange x and y
    centers = centers.tolist()
    centers = [coord[::-1] for coord in centers]
    centers = np.array(centers)

    if visu and centers.size != 0:
        plt.figure(figsize=(6, 6))
        plt.gca().invert_yaxis()
        plt.scatter(total_points[:, 1], total_points[:, 0], c=model.labels_)
        plt.scatter(centers[:, 0], centers[:, 1], s=20, c=["orange"]*len(centers))
        plt.scatter(gt_points[:, 0], gt_points[:, 1], s=20, c=["red"]*len(gt_points))
        plt.title("Predicted_blobs")
        plt.tight_layout()
        plt.savefig(rslt_path1+".png")
        plt.close()
        
    return centers, len(centers), total_time

##############################################
#       Gaussian Mixture Models (GMN)        #
##############################################

def compute_wgmn_ski(out,im_path,resultsPath,gt_points,visu=False):
    """
    Computes the Gaussian Mixture Models (GMM) using the sklearn library for the given output and ground truth points.
    Args:
        out (torch.Tensor): The output density map. size: (H,W)
        im_path (str): Path of the image, used to get the name of the image.
        resultsPath (str): Directory path to save the results.
        gt_points (np.ndarray): Ground truth points. size: (nb_points,2)
        visu (bool): Whether to visualize the results or not.
    Returns:
        centers (np.ndarray): The centers of the clusters detected by GMM. size: (nb_clusters,2)
        len(centers) (int): The number of clusters detected.
        total_time (float): The total time taken to compute the GMM.
    """
    image_name = str(Path(im_path).stem)
    rslt_path1 = os.path.join(resultsPath, image_name + "_gmm_ski")

    pred_count = round(out.sum().item())

    start = time.time()
    points, _, points_bool = den2points(out)

    #weighting
    total_points = points.tolist()
    max_value = round(torch.max(out).item(),4)
    range_of_value = [round(max_value/10,4)*i for i in range(10)]

    for h in range(out.shape[0]):
        for w in range(out.shape[1]):
            if points_bool[h][w] == 1:
                for ind, value in enumerate(range_of_value):
                    if ind != len(range_of_value)-1:
                        if out[h][w] > value and out[h][w] < range_of_value[ind+1] and ind != 0:
                            total_points = total_points + [[h,w]]*ind
                    else:
                        if out[h][w] > value:
                            total_points = total_points + [[h,w]]*ind

    total_points = np.array(total_points)

    #gmm models
    gmm = GaussianMixture(n_components=pred_count, max_iter=20).fit(total_points)

    centers = np.empty(shape=(gmm.n_components, 2))
    for i in range(gmm.n_components):
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i],allow_singular=True).logpdf(total_points)
        centers[i, :] = total_points[np.argmax(density)]
    end = time.time()
    total_time = round(end-start,3)
    print("Total time to compute GMM ski: ",str(total_time)," s")

    # exchange x and y
    centers = centers.tolist()
    centers = [coord[::-1] for coord in centers]
    centers = np.array(centers)

    if visu:
        plt.figure(figsize=(6, 6))
        plt.gca().invert_yaxis()
        plt.scatter(points[:, 1], points[:, 0], c=gmm.predict(points),s=2)
        plt.scatter(centers[:, 0], centers[:, 1], s=20, c=["orange"]*len(centers))
        plt.scatter(gt_points[:, 0], gt_points[:, 1], s=20, c=["red"]*len(gt_points))
        plt.title("Predicted_blobs")
        plt.tight_layout()
        plt.savefig(rslt_path1+".png")
        plt.close()



    return centers, len(centers), total_time  

def den2points(out):
    """
    Converts a density map to a list of points and their corresponding weights. 
    Args:
        out (torch.Tensor): The output density map. size: (H,W)
    Returns:
        points (np.ndarray): The list of points where the density is above a threshold. size: (nb_points,2)
        weights (np.ndarray): The weights corresponding to the points. size: (nb_points,)
        points_bool (np.ndarray): A boolean mask indicating the points where the density is above a threshold. size: (H,W)
    """
    points = []
    weights = []
    points_bool = np.zeros_like(out)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if out[i][j] > 0.0001:
                points.append([i,j])
                weights.append(out[i][j])
                points_bool[i][j]=1
            else:
                points_bool[i][j]=0


    return np.array(points), np.array(weights), points_bool


def compute_min_size_object(boxes):
    """
    Computes the minimum size of the objects in the bounding boxes.
    Args:
        boxes (torch.Tensor): Bounding boxes of the objects. size: (nb_boxes, 4)
    Returns:
        min_size (int): The minimum size of the objects in the bounding boxes.
    """
    min_size = 384
    for j in range(0, boxes.shape[0]):
        x1, y1, x2, y2 = int(boxes[j, 0].item()), int(boxes[j, 1].item()), int(boxes[j, 2].item()), int(boxes[j, 3].item())
        min_size_temp = min(x2 - x1, y2 - y1)
        min_size = min_size_temp if min_size_temp < min_size else min_size

    return min_size


def hungarian_matching(out, gt_centroids, min_size_object,im_path, resultsPath, patch_size, kernel_dim):
    """
    Computes the Hungarian matching between the predicted centroids and the ground truth centroids.
    This function computes the precision, recall, and F1 score of the predicted centroids compared to the ground truth centroids.
    Args:
        out:             torch tensor of size [H,W]
        gt_centroids:    np.ndarray of size [nb_points,2]
        min_size_object: minimal with or height of the bounding boxes
        im_path:         path of the image, used to get the name of the image
        resultPath:      directory path
        patch_size:      size of the patch used to compute the density map
        kernel_dim:      size of the kernel used to compute the density map
    Returns:
        None: The results are saved in the resultsPath with a csv file named "hm.csv" containing the precision, recall, F1 score, and computation time for each method.

    """
    # saving file
    image_name = str(Path(im_path).stem)
    rslt_path = os.path.join(resultsPath, "hm.csv")
    columns_name = ["image_name","method","patch_size", "kernel_dim","precision","recall","f1_score","computation_time"]

    if (os.path.exists(rslt_path)):
        df = pd.read_csv(rslt_path)
        if image_name in df["image_name"].values:
            return
    else:
        df = pd.DataFrame(columns=columns_name)

    centroids_otm, _, time_otm         = compute_otm(out,gt_centroids,visu=False)
    #centroids_gmns,_, time_gmns        = compute_wgmn_ski(out,im_path,resultsPath,gt_centroids,visu=False)
    #centroids_hdbscan,_, time_hdbscan  = compute_hdbscan(out,im_path,resultsPath,min_size_object,gt_centroids,visu=False)

    #list_of_centroids = {"otm": centroids_otm, "gmms": centroids_gmns, "hdbscan": centroids_hdbscan} #",int_prog": centroids_int_prog}
    list_of_centroids = {"otm": centroids_otm}
    #list_of_time = {"otm": time_otm, "gmms": time_gmns, "hdbscan": time_hdbscan} #",int_prog": centroids_int_prog}
    list_of_time = {"otm": time_otm}
    

    for method, pred_centroids in list_of_centroids.items():

        computation_time = list_of_time[method]

        if pred_centroids.size == 0:
            results = [image_name, method, patch_size, kernel_dim] + [0,0,0,computation_time]
            row_to_add = pd.DataFrame([results],columns=columns_name)
            df  = pd.concat([df, row_to_add], ignore_index=True)

            print("Results for image ",image_name," with the method ",method,": P: ",0," R: ",0," F1: ",0)
            continue

        # RECALL
        distances = np.linalg.norm(gt_centroids[:, np.newaxis] - pred_centroids, axis=2) #compute 1-1 distances
        row_ind, col_ind = linear_sum_assignment(distances)                              #compute the optimal assignement via Hungarian algorithm, distances is the cost matrix

        matches = list(zip(row_ind, col_ind))
        recall = 0
        for i, j in matches:
            #print(f"GT centroid {i} matched with predicted centroid {j}, distance: {distances[i, j]}")
            if distances[i,j] < min_size_object:
                recall += 1
        recall = recall/len(gt_centroids)

        # PRECISION
        row_ind, col_ind = linear_sum_assignment(distances.T)
        matches = list(zip(row_ind, col_ind))
        
        precision = 0
        for i, j in matches:
            #print(f"predicted centroid {i} matched with GT centroid {j}, distance: {distances[j, i]}")
            if distances[j,i] < min_size_object:
                precision += 1
        precision = precision/len(pred_centroids)
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = round((2*precision*recall) / (precision+recall),3)

        results = [image_name, method, patch_size, kernel_dim] + [round(precision,3),round(recall,3),f1, computation_time]
        row_to_add = pd.DataFrame([results],columns=columns_name)
        df  = pd.concat([df, row_to_add], ignore_index=True)

        print("Results for image ",image_name," with the method ",method,": P: ",round(precision,3)," R: ",round(recall,3)," F1: ",f1)
    
    print("\n--- next image ---\n")

    df.to_csv(rslt_path, index=False)

    return


#-------------------------------------------------------------#

# Script for saving statistics about the results

#-------------------------------------------------------------#

def save_data_statistics(out,gt,bboxes,im_path,cnt_err, relative_error,resultsPath,image_size,patch_size,kernel_dim):
    """
    TODO
    """

    widths = []
    heights = []
    current_bboxes = bboxes.detach().cpu().squeeze(0)
    current_output = out.detach().cpu().unsqueeze(0)
    for j in range(0, current_bboxes.shape[0]):
        x1, y1, x2, y2 = int(current_bboxes[j, 0].item()), int(current_bboxes[j, 1].item()), int(current_bboxes[j, 2].item()), int(current_bboxes[j, 3].item())
        widths.append(x2 - x1)
        heights.append(y2 - y1)

    diff_maps = current_output - (gt.detach().cpu().unsqueeze(0))
    diff_maps = torch.maximum(diff_maps,torch.tensor(0))
    sum_diff_maps = diff_maps.sum().item()
    percentage_irrelevance = round(sum_diff_maps/(out.detach().cpu().unsqueeze(0).sum().item()),2)
    
    c_name = ["image_name", "resized_size", "patch_size", "kernel_dim","Avg_ref_h", "Avg_ref_z", "mean_box_size", "std_size", "diff_patch_mbs", "aspect_ratio", "mae", "mre","qual_score"]
    if os.path.exists(os.path.join(resultsPath, "data_statistics.csv")):
        df = pd.read_csv(os.path.join(resultsPath, "data_statistics.csv"))
    else:
        df = pd.DataFrame(columns=c_name)

    #widths = [int(box[2].item())-int(box[0].item()) for box in bboxes[i].detach().cpu().squeeze(0)]
    #heights = [int(box[3].item())-int(box[1].item()) for box in bboxes[i].detach().cpu().squeeze(0)]

    std_width = np.std(widths)
    std_height = np.std(heights)

    avg_width = round(np.average(widths))
    avg_height = round(np.average(heights))

    aspect_ratio = round(avg_height/avg_width,2) if avg_height > avg_width else round(avg_width/avg_height,2)

    mean_size = round((avg_width + avg_height)/2)
    mean_std = round((std_width + std_height)/2,2)

    # std -> translate the variance in size of the objects in the image
    # aspect ratio -> translate how far the object is rectangular
    # diff -> translate how much the object is resized

    if "Image" in str(Path(im_path).stem):
        image_name = str(Path(im_path).stem)
    else:
        image_name = "Image_" +  str(Path(im_path).stem)

    diff = abs(patch_size - mean_size)
    df2 = pd.DataFrame([[image_name, image_size, patch_size, kernel_dim, avg_width, avg_height, mean_size, mean_std, diff, aspect_ratio, cnt_err, relative_error,percentage_irrelevance]], columns=c_name)
    df  = pd.concat([df2, df], ignore_index=True)

    df.to_csv(os.path.join(resultsPath, "data_statistics.csv"), index=False)

#----------------------------------------------------------------------------------#

# Optimal transport algorithm

#----------------------------------------------------------------------------------#

# -*- coding: utf-8 -*-
#from https://github.com/Elin24/OT-M/blob/main/simple_ot.py

import torch
import torch.nn.functional as tF
import numpy as np

EPS =  1e-12

def max_diameter(x, y):
    mins = torch.stack((x.min(dim=1)[0], y.min(dim=1)[0]), dim=1).min(dim=1)[0] # B 2
    maxs = torch.stack((x.max(dim=1)[0], y.max(dim=1)[0]), dim=1).max(dim=1)[0] # B 2
    diameter = (maxs-mins).norm(dim=1).max().item()
    if diameter == 0:
        diameter = 16
    return diameter

def epsilon_schedule(diameter, blur, scaling, fixed_epsilon=False):
    # print("[EPS]:", np.log(diameter), np.log(blur), np.log(scaling))
    schedule = np.arange(np.log(diameter), np.log(blur), np.log(scaling))
    if fixed_epsilon:
        epsilon_s = [ blur ] + [ blur for _ in  schedule] + [ blur ]
    else:
        epsilon_s = [ diameter ] + [ np.exp(e) for e in schedule ] + [ blur ]
    return epsilon_s

def dampening(epsilon, reach):
    return 1 if reach is None else 1 / ( 1 + epsilon / reach )

def softmin(logB, G, C, epsilon):
    B = C.shape[0]
    x = logB.view(B, 1, -1) + (G.view(B, 1, -1) - C) / epsilon
    x = - epsilon * x.logsumexp(2).view(B, -1, 1)
    return x

class SampleOT:
    def __init__(self, blur=0.01, scaling=0.5, reach=None, fixed_epsilon=False) -> None:
        self.blur = blur
        self.scaling = scaling
        self.fixed_epsilon = fixed_epsilon
        self.reach = reach

    @torch.no_grad()        
    def __call__(self, A, B, cost, F=None, G=None, diameter=None):
        '''
        A.shape = B H 1
        B.shape = B W 1
        cost.shape = B H W
        '''
        
        bsize, H, W = cost.shape
        
        fixed_epsilon = (F is not None and G is not None) or self.fixed_epsilon
        diameter = diameter if diameter is not None else cost.max().item()
        diameter = max(8, diameter)
        epsilons = epsilon_schedule(diameter, self.blur, self.scaling, fixed_epsilon)
        
        logA, logB = A.log(), B.log()
        Cab, Cba = cost, cost.permute(0, 2, 1)
        factor = dampening(epsilons[0], self.reach)
        if F is None:
            F = factor * softmin(logB, torch.zeros_like(B), Cab, epsilons[0])
        if G is None:
            G = factor * softmin(logA, torch.zeros_like(A), Cba, epsilons[0])
            
        for i, epsilon in enumerate(epsilons):

            factor = dampening(epsilon, self.reach)
            tF = factor * softmin(logB, G, Cab, epsilon)
            tG = factor * softmin(logA, F, Cba, epsilon)
            F, G = (F + tF) / 2, (G + tG) / 2

        factor = dampening(self.blur, self.reach)
        F, G = factor * softmin(logB, G, Cab, self.blur), factor * softmin(logA, F, Cba, self.blur)
        

        return F.detach(), G.detach()

    def loss(self, A, B, F, G):
        if self.reach is not None:
            F = self.weightfunc(1 - (- F / self.reach).exp())
            G = self.weightfunc(1 - (- G / self.reach).exp())
        return torch.mean( (A * F).flatten(1).sum(dim=1) + (B * G).flatten(1).sum(dim=1) )


    def plan(self, A, B, F, G, cost):
        PI1 = torch.exp((F + G.permute(0, 2, 1) - cost) / self.blur)
        PI2 = A * B.permute(0, 2, 1)
        PI = PI1 * PI2
        return PI

class L2_DIS:
    factor = 1 / 32
    @staticmethod
    def __call__(X, Y):
        '''
        X.shape = (batch, M, D)
        Y.shape = (batch, N, D)
        returned cost matrix's shape is ()
        '''
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        C = ((x_col - y_row) ** 2).sum(dim=-1) / 2
        return C * L2_DIS.factor

    @staticmethod
    def barycenter(weight, coord):
        '''
        weight.shape = (batch, M, N)
        coord.shape = (batch, M, D)
        returned coord's shape is (batch, N D)
        '''
        weight = weight / (weight.sum(dim=1, keepdim=True) + EPS)
        return weight.permute(0, 2, 1) @ coord

blur = 0.01
per_cost = L2_DIS()
ot = SampleOT(blur=blur, scaling=0.9, reach=None, fixed_epsilon=False)

def den2coord(denmap, scale_factor=8):
    coord = torch.nonzero(denmap > EPS)
    denval = denmap[coord[:, 0], coord[:, 1]]
    if scale_factor != 1:
        coord = coord.float() * scale_factor + scale_factor / 2
    coord = coord.to(torch.float)
    return denval.reshape(1, -1, 1), coord.reshape(1, -1, 2)

def init_dot(denmap, n, scale_factor=8):

    norm_den = denmap[None, None, ...]
    norm_den = tF.interpolate(norm_den, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    norm_den = norm_den[0, 0]

    d_coord = torch.nonzero(norm_den > EPS)
    norm_den = norm_den[d_coord[:, 0], d_coord[:, 1]]

    cidx = torch.multinomial(norm_den, num_samples=n, replacement=False)
    coord = d_coord[cidx]
    
    B = torch.ones(1, n, 1).to(denmap)
    B_coord = coord.reshape(1, n, 2)
    return B, B_coord

@torch.no_grad()
def OT_M(A, A_coord, B, B_coord, scale_factor=8, max_itern=8):
    for iter in range(max_itern):
        # OT-step
        C = per_cost(A_coord, B_coord)
        F, G = ot(A, B, C)
        PI = ot.plan(A, B, F, G, C)
        # M-step
        nB_coord = per_cost.barycenter(PI, A_coord)
        move = torch.norm(nB_coord - B_coord, p=2, dim=-1)
        if move.mean().item() < 1 and move.max().item() < scale_factor:
            break
        B_coord = nB_coord
    
    return (nB_coord).reshape(-1, 2)

@torch.no_grad()
def den2seq(denmap, scale_factor=8, max_itern=16, ot_scaling=0.75):
    ot.scaling = ot_scaling
    assert denmap.dim() == 2, f"the shape of density map should be [H, W], but the given one is {denmap.shape}"
    
    num = int(denmap.sum().item() + 0.5)
    if num < 0.5:
        return torch.zeros((0, 2)).to(denmap)

    # normalize density map
    denmap = denmap * num / denmap.sum()
    
    A, A_coord = den2coord(denmap, scale_factor)
    B, B_coord = init_dot(denmap, num, scale_factor)

    flocs = OT_M(A, A_coord, B, B_coord, scale_factor, max_itern=max_itern)
    return flocs





    
