from pathlib import Path
from PIL import Image
from os.path import dirname, abspath
import os
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
import cv2

rootDirectory = dirname(abspath(__file__))

################################################################################

# MAIN PROCEDURE

# This python file aims to create the FSCEuresys dataset for the AR comparison.
# It will crop the images into 384x384 images, and measure the size of the object in the image.
# If the object is too small, it will concat the images to make the object bigger.
# If the object is too big, it will crop the image to make the object smaller.

# -----------------------------------------------------------------------------#

# Helpers

drawing = False
width_and_height = []
mode = "rectangle"

def draw_bbox(event, x,y, flags, param):

    global x1, y1, drawing, image, image2,width_and_height, OUTPUT_IMAGE

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
            width_and_height = [x2-x1, y2-y1]
            object_size = str(round(np.sqrt(width_and_height[0]*width_and_height[1]),2))
            cv2.putText(image,text="Object Size:  "+object_size,org=(x,y),color=(0,0,0),fontFace=cv2.FONT_HERSHEY_SIMPLEX,lineType=cv2.LINE_AA, fontScale=0.75, thickness=4)
            cv2.putText(image,text="Object Size:  "+object_size,org=(x,y),color=(0,255,0),fontFace=cv2.FONT_HERSHEY_SIMPLEX,lineType=cv2.LINE_AA, fontScale=0.75, thickness=1)
            image2 = image.copy()
            #cv2.imwrite(OUTPUT_IMAGE, image2)

        else:
            cv2.circle(image,(x,y),5,(0,0,255),1)

def concat4images(Pil_image):
    w,h = Pil_image.size
    back_ground = Image.new("RGB", (2*w,2*h))

    #paste the four images
    back_ground.paste(Pil_image, (0,0))
    back_ground.paste(Pil_image, (w,0))
    back_ground.paste(Pil_image, (0,h))
    back_ground.paste(Pil_image, (w,h))

    return back_ground

def concat2images(Pil_image):
    w,h = Pil_image.size
    back_ground = Image.new("RGB", (w,2*h))

    #paste the four images
    back_ground.paste(Pil_image, (0,0))
    back_ground.paste(Pil_image, (0,h))

    return back_ground

def concat_images(Pil_image, concat_ratio):
    w,h = Pil_image.size
    # we concat on the lowest dimension
    if w > h :
        background = Image.new("RGB", (w,int(concat_ratio*h)))
        concat_im = fill_background(Pil_image,background)
    else:
        background = Image.new("RGB", (int(concat_ratio*w),h))
        concat_im  = fill_background(Pil_image,background)

    return concat_im

def fill_background(Pil_image,background):
    w,h = Pil_image.size
    w_b, h_b = background.size
    w_range = w_b // w
    h_range = h_b // h
    for i in range(w_range+1):
        for j in range(h_range+1):
            background.paste(Pil_image, (i*w,j*h))
    
    return background
    


# -----------------------------------------------------------------------------#

def all_folder(folder_path,required_range,ratio,centered_cropping=False):

    last_folder_name = Path(folder_path).stem
    parent_folder = Path(folder_path).parent
    folder_for_appropriate_range = os.path.join(parent_folder, str(last_folder_name) + "_" + str(required_range[1]) +"_"+ratio)

    if not os.path.exists(folder_for_appropriate_range):
        os.mkdir(folder_for_appropriate_range)

    for file in os.listdir(folder_path):
        fp = os.path.join(folder_path,file)

        im_id = Path(fp).stem
        if os.path.exists(os.path.join(folder_for_appropriate_range,str(im_id)+".png")):
            continue

        object_size = -1
        concat_ratio = 1
        resized_image = None
        rim_path = ""

        while object_size < required_range[0] or object_size > required_range[1] or object_size == -1:
            resized_image, rim_path = crop_into_384_images(fp,folder_for_appropriate_range,concat_ratio)

            resized_image.save(rim_path)
            #annotated_image, object_size = measure_object_size(rim_path, folder_for_appropriate_range)
            object_size = measure_object_size(rim_path, folder_for_appropriate_range)
            

            #save only the image if the size of the object is appropriate
            if object_size < required_range[0]:
                min_zoom = object_size/required_range[0] # for a  "real" zoom, we should have put 1/(object_size/required_range[0]) to have value > 1...
                best_zoom = object_size/((required_range[1]-required_range[0])/2)

                if object_size/min_zoom> required_range[1]:
                    concat_ratio = best_zoom
                else:
                    concat_ratio = min_zoom

            if object_size > required_range[1]:
                
                min_ratio = object_size/required_range[1]
                best_ratio = (required_range[1]-required_range[0])/2
                best_ratio_int = np.ceil(min_ratio)

                if object_size/best_ratio_int < required_range[0]:
                    concat_ratio = best_ratio
                else:
                    concat_ratio = best_ratio_int

            fp = rim_path

        orig_h, orig_w = resized_image.size
        cropped_image = Image.new("RGB", (384,384))

        # crop the image at the center, and not at the upper left of the image
        if centered_cropping:
            if orig_h > 384:
                starting = int((orig_h - 384)/2)
                cropped_image.paste(Image.fromarray(np.array(resized_image)[:,starting:,:]), (0,0))
            else:
                starting = int((orig_w - 384)/2)
                cropped_image.paste(Image.fromarray(np.array(resized_image)[starting:,:,:]), (0,0))
        else:
            cropped_image.paste(resized_image, (0,0))

        cropped_image.save(rim_path)

        fig, (ax1,ax2,ax3) = plt.subplots(1,3)

        ax1.imshow(Image.open(fp).convert("RGB"))
        ax2.imshow(resized_image)
        #ax3.imshow(annotated_image)

        plt.tight_layout()
        plt.show()


def crop_into_384_images(image_path, new_image_folder,concat_ratio):

    im_id = Path(image_path).stem
    im = Image.open(image_path).convert("RGB")
    im.load()

    # concat 2 image to get smaller object
    im = concat_images(im,concat_ratio)

    orig_w, orig_h = im.size

    print("Original width: ", orig_w, " Original height: ", orig_h)

    if not os.path.exists(new_image_folder):
        os.mkdir(new_image_folder)

    if orig_w < orig_h :
        resize_ratio = 384/orig_w
        new_width = 384
        new_height = int(resize_ratio * orig_h)
    else:
        resize_ratio = 384/orig_h
        new_width = int(resize_ratio * orig_w)
        new_height = 384

    resized_image = transforms.Resize((new_height, new_width))(im)
    path = os.path.join(new_image_folder,str(im_id)+".png")

    return resized_image, path

def measure_object_size(im_path, new_image_folder):

    global image, image2,width_and_height, OUTPUT_IMAGE

    width_and_height = []

    # Define the file name of the image
    im_id = Path(im_path).stem
    INPUT_IMAGE  = im_path
    OUTPUT_IMAGE = os.path.join(new_image_folder,str(im_id) + "_annotated.png")
    
    # Load the image and store into a variable
    image = cv2.imread(INPUT_IMAGE, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = image.copy()

    cv2.namedWindow('Image mouse')
    cv2.setMouseCallback('Image mouse', draw_bbox)

    while True:

        # Show image 'Image mouse':
        cv2.imshow('Image mouse', image)
    
        # Continue until 'q' is pressed:
        if (cv2.waitKey(20) & 0xFF == ord('q')):
            #cv2.imwrite(OUTPUT_IMAGE,image2)
            break

    object_size = np.sqrt(width_and_height[0]*width_and_height[1])
    print("Object size: ", object_size)

    cv2.destroyAllWindows()

    return object_size


image_folder_path = r"c:\Users\bourezn\Documents\Master_thesis\data\Image_orin\Small_nails2\images"

all_folder(image_folder_path, required_range=(28,45), ratio="23")


# draft

"""
div_ent_h = h//384
    rest_h = h - div_ent_h*384
    step_h = 384 - rest_h/div_ent_h

    div_ent_w = w//384
    rest_w = w - div_ent_w*384
    step_w = 384 - rest_w/div_ent_w

    im_id=0
    for i in range(div_ent_h):
        for j in range(div_ent_w):
            im_id+=1
            height = 384
            width = 384
            x1,y1 = j*step_h,i*step_w
            x2,y2 = min(384,x1 + width), min(384,y1 + height)
            cropped_im = Image.fromarray(im[:,y1:y2,x1:x2])
            cropped_im.save(os.path.join(new_image_folder,"Image_"+str(im_id)+".png"))

            x2 = x
            y2 = y
            width_and_height = [x2-x1, y2-y1]
            object_size = str(np.sqrt(width_and_height[0]*width_and_height[1]))
            cv2.putText(image,text="Object Size:  "+object_size,org=(x,y),color=(0,255,0),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5, thickness=3)
"""

# take a picture
# if object are not small enough -> concat the images
# resize to width or height = 384 by keeping the aspect ratio of the (concatened) image
# crop the image to get a 384x384 -> wee keep the aspect ratio of the object while resizing the image at the specified dimension




            

