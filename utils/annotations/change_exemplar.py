#################
# Author : Nicolas Bourez
#
# date: 28/01/2025
#######################

# Imports


import json
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import pandas as pd # Import Pandas library
import sys # Enables the passing of arguments
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from pathlib import Path
import os

import argparse
import time
from os.path import dirname, abspath

#################################################################

# Same file as the script_annotation but changing the bounding boxes
# no dot annotation here, only bounding boxes

#################################################################

drawing = False
nb_bbox = 0

rootDirectory = dirname(abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-image", type=str, required=True, help="/Path/to/input/image/file/")

points = []
box_all_coordinates = []
box_coordinates = []
mode = "rectangle"

def creating_folders(dataset_path):
    # create the directory if it does not exist yet
    if not os.path.exists(os.path.join(dataset_path, "images_annotated")):
        os.makedirs(os.path.join(dataset_path, "images_annotated"))
    if not os.path.exists(os.path.join(dataset_path, "gt_density_map")):
        os.makedirs(os.path.join(dataset_path, "gt_density_map"))
    if not os.path.exists(os.path.join(dataset_path, "bboxes")):
        os.makedirs(os.path.join(dataset_path, "bboxes"))

    return

def gaussian_filter_density(img,points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

    return:
    density: the density-map we want. Same shape as input image but only has one channel.

    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    print("Totally need generate",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, _ = tree.query(points, k=2)
    print("Results from KNN: \n")
    print("Nombre de distances: ",len(distances))
    radius = round(np.average(distances))
    print("Radius: ", radius)
    sigma = round(radius/4)
    print("Sigma :", sigma)

    print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
    
        density += scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant',radius=radius)

    print ('done.')
    #print(density.sum())
    return density

############################################################
# DRAWING FUNCTIONS
############################################################

def draw_bbox(event, x,y, flags, param):

    global x1, y1, drawing, nb_bbox, OUTPUT_IMAGE, image, image2

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1,y1 = x,y
        if mode == "rectangle":
            cv2.rectangle(image,(x1,y1),(x,y),(0,255,0),1)
        else:
            cv2.circle(image,(x,y),5,(0,0,255),1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            a = x
            b = y
            if a != x & b != y:
                image = image2.copy()
                if mode == "rectangle":
                    cv2.rectangle(image,(x1,y1),(x,y),(0,255,0),1)
                else:
                    cv2.circle(image,(x,y),5,(0,0,255),1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == "rectangle":
            cv2.rectangle(image,(x1,y1),(x,y),(0,255,0),1)
            x2 = x
            y2 = y
            box_coordinates.append([y1, x1, y2, x2])
            box_all_coordinates.append([[x1,y1],[x1,y2],[x2,y2],[x2,y1]])
            nb_bbox += 1
            image2 = image.copy()
            #cv2.imwrite(OUTPUT_IMAGE, image2)
            
            if len(box_coordinates) == 3:
                print("You have now enough rectangles")
            elif len(box_coordinates) == 2:
                print("Rectangle " + str(len(box_coordinates)) + " drawed, " + str(3 - len(box_coordinates)) + " more is needed")
            else:
                print("Rectangle " + str(len(box_coordinates)) + " drawed, " + str(3 - len(box_coordinates)) + " more are needed")


        else:
            cv2.circle(image,(x,y),5,(0,0,255),1)

def main(im_path,counter):

    start = time.time()
    global image, image2, OUTPUT_IMAGE, nb_bbox, points, box_coordinates, box_all_coordinates

    points = []
    box_all_coordinates = []
    box_coordinates = []

    # Define the file name of the image
    INPUT_IMAGE = im_path

    print(INPUT_IMAGE)

    dataset_path = Path(INPUT_IMAGE).parent.parent
    dataset_name = dataset_path.stem
    dataset_name = dataset_name.replace("images",dataset_path.parent.stem)

    creating_folders(dataset_path)
    print(str(dataset_path))
    #INPUT_IMAGE = "Images_0.png"
    IMAGE_NAME = Path(INPUT_IMAGE).stem
    whole_image_name = os.path.basename(INPUT_IMAGE)
    OUTPUT_IMAGE = os.path.join(dataset_path, "images_annotated/" + IMAGE_NAME + "_32_annotated.jpg")

    #check if the image has already been processed
    if os.path.exists(OUTPUT_IMAGE):
        print("The file exists.")
        return
    
    # Load the image and store into a variable
    image = cv2.imread(INPUT_IMAGE, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = image.copy()

    ############################################################
    # DRAWING
    ############################################################
 
    # Prompt user for another annotation
    print("Welcome to the Image Annotation Program!\n")
    print("Double click anywhere inside the image to annotate that point...\n")
    
    # We create a named window where the mouse callback will be established
    cv2.namedWindow('Image mouse',cv2.WINDOW_FULLSCREEN)

    print("Draw the bounding boxes of your 3 examples")

    cv2.setMouseCallback('Image mouse', draw_bbox)

    while True:

        # Show image 'Image mouse':
        cv2.imshow('Image mouse', image)
    
        # Continue until 'q' is pressed:
        if (cv2.waitKey(20) & 0xFF == ord('q')) or nb_bbox == 3:
            break

    nb_bbox = 0

    ############################################################
    # SAVING ANNOTATIONS
    ############################################################

    #generating bbox_file
    out_bbox_file = os.path.join(dataset_path, "bboxes/"+ IMAGE_NAME + "_32_box.txt")
    fout = open(out_bbox_file, "w")
    rects1 = list()
    for rect in box_coordinates:
        x1, y1, x2, y2 = rect
        rects1.append([x1, y1, x2, y2])
        fout.write("{} {} {} {}\n".format(x1, y1, x2, y2))

    fout.close()

    #add to json file
    if counter == 0:
        json_file = os.path.join(dataset_path, "annotation.json")
        if os.path.exists(json_file):
            with open(json_file, 'r+') as f:
                annotations = json.load(f)
        else:
            raise Exception("Need to run script_annotation_all_dataset.py before !!!")
    else:
        json_file = os.path.join(dataset_path, "annotation_32.json")
        if os.path.exists(json_file):
            with open(json_file, 'r+') as f:
                annotations = json.load(f)
        else:
            raise Exception("Need to run script_annotation_all_dataset.py before !!!")

    annotations[whole_image_name]["box_examples_coordinates"] = box_all_coordinates
    annotations[whole_image_name]["box_coordinates"] = box_coordinates

    new_json_file = os.path.join(dataset_path, "annotation_32.json")
    with open(new_json_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    
    end = time.time()

    print("Annotation time: " + str(round(end-start)) + " seconds")
    
    # Destroy all generated windows:
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input-dir", type=str, default= r"C:\Users\bourezn\Documents\Master_thesis\data\Image_orin\Nails_23\images_384")

    args = parser.parse_args()
    dir_path = args.input_dir
    dir_path = os.path.join(dir_path, "images\\")

    files =  [os.path.abspath(os.path.join(dir_path,f)) for f in os.listdir(dir_path)]

    for i,f in enumerate(files):
        if ".png" not in f and ".jpg" not in f:
            raise Exception("Your directory does not contain any image")
        else:
            main(f,i)


"""
if (bbox_file is None or bbox_file == ""): # if no bounding box file is given, prompt the user for a set of bounding boxes
        out_bbox_file = "{}/{}_box.txt".format(rootDirectory, "utils/" + image_name)
        fout = open(out_bbox_file, "w")

        im = cv2.imread(input_image)
        cv2.imshow('image', im)
        rects = select_exemplar_rois(im)

        rects1 = list()
        for rect in rects:
            y1, x1, y2, x2 = rect
            rects1.append([y1, x1, y2, x2])
            fout.write("{} {} {} {}\n".format(y1, x1, y2, x2))

        fout.close()
        cv2.destroyWindow("image")
        bbox_file = out_bbox_file
        print("selected bounding boxes are saved to {}".format(out_bbox_file))
"""
    