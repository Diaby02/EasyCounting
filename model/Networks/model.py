from matplotlib import pyplot as plt
from utils.utils_ import ObjectNormalizedL2Loss, getInfoModel, createFolder
from utils.dataLoader import *
from .Architectures.loca import build_model
from utils.visu import partial_vizu
from utils.utils_ import *
from ptflops import get_model_complexity_info
import duckdb as db
import re

import numpy as np
np.random.seed(2885)
import os

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import numpy as np

from tqdm import tqdm

######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# An instance of Network_Class has been created in the main.py file
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # INPUTS: 
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - resultsPath (str): path to the folder containing the results of the 
    #                          experiement
    # --------------------------------------------------------------------------------
    def __init__(self, param, resultsPath, checkpointsPath=None):
        # ----------------
        # USEFUL VARIABLES 
        # ----------------
        
        self.param         = param
        self.resultsPath   = resultsPath
        self.checkpoints   = checkpointsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.backbone_lr   = param["TRAINING"]["BACKBONE_LR"]
        self.lr_drop       = param["TRAINING"]["LR_DROP"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]
        self.weight_decay  = param["TRAINING"]["WEIGHT_DECAY"]
        self.dropout       = param["TRAINING"]["DROPOUT"]
        self.max_grad_norm = param["TRAINING"]["MAX_GRAD_NORM"]
        self.aux_weight    = param["TRAINING"]["AUX_WEIGHT"]
        self.prenorm       = param["TRAINING"]["PRENORM"]

        self.num_objects             = param["MODEL"]["NUM_OBJECTS"]
        self.kernel_dim              = param["MODEL"]["KERNEL_DIM"]
        self.num_decoder_layers      = param["MODEL"]["NUM_DECODER_LAYERS"]
        self.num_ope_iterative_steps = param["MODEL"]["NUM_OPE_ITERATIVE_STEPS"]
        self.padding                 = param["MODEL"]["PADDING"]

        self.device        = torch.device(param["TRAINING"]["DEVICE"])

        # -----------------------------------
        # DATASET ATTRIBUTES
        # -----------------------------------

        self.data_path        = param["DATASET"]["DATA_PATH"]
        self.image_size       = param["DATASET"]["IMAGE_SIZE"]
        self.patch_size       = param["DATASET"]["PATCH_SIZE"]
        self.patch_size_ratio = param["DATASET"]["PATCH_SIZE_RATIO"]
        self.tiling_p         = param["DATASET"]["TILING_P"]
        self.image_folder     = param["DATASET"]["IMAGE_FOLDER"]
        self.gt_folder        = param["DATASET"]["GT_FOLDER"]
        self.split_file       = param["DATASET"]["SPLIT_FILE"]
        self.annotation_file  = param["DATASET"]["ANNOTATION_FILE"]
        
        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        
        self.model = build_model(param).to(self.device)
        
        # -------------------
        # TRAINING PARAMETERS
        # -------------------

        # separating the parameters of the model
        if param["TRAINING"]["TRAIN"] != False:
            self.saving_name   = param["MODEL"]["SAVING_NAME"]
            self.training_data = param["DATASET"]["TRAINING_DATA"]
            backbone_params = dict()
            other_params = dict()
            for name, parameters in self.model.named_parameters():
                if not parameters.requires_grad:
                    continue
                if 'backbone' in name:
                    backbone_params[name] = parameters
                else:
                    other_params[name] = parameters

            self.criterion = ObjectNormalizedL2Loss()
            self.optimizer = torch.optim.Adam([
                {'params': backbone_params.values(), 'lr': self.backbone_lr},
                {'params': other_params.values(), 'lr': self.lr}
            ], weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_drop, gamma=0.25)

        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
            self.dataSetTrain    = FSCDataset(self.data_path, self.image_size, self.patch_size, self.image_folder, self.gt_folder, 
                                              self.split_file, self.annotation_file, self.training_data, self.num_objects, self.tiling_p, 
                                              padding=self.padding, patch_size_ratio=self.patch_size_ratio)
            self.dataSetVal      = FSCDataset(self.data_path, self.image_size, self.patch_size, self.image_folder, self.gt_folder, 
                                              self.split_file, self.annotation_file, 'val',   self.num_objects, self.tiling_p, 
                                              padding=self.padding, patch_size_ratio=self.patch_size_ratio)
            self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=4)
            self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=4)
        
        if param["TRAINING"]["EVALUATE"]:
            self.model_name    = param["MODEL"]["MODEL_NAME"]
            self.test_data     = param["DATASET"]["TEST_DATA"]
            self.model_path    = param["MODEL"]["MODEL_PATH"]
            self.dataSetTest = FSCDataset(self.data_path, self.image_size, self.patch_size, self.image_folder, self.gt_folder,
                                           self.split_file, self.annotation_file, self.test_data, self.num_objects, self.tiling_p, 
                                           padding=self.padding, patch_size_ratio=self.patch_size_ratio)
            self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=4)



    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
        if 'full' in self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)['model'] # load the model
            self.model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(self.model_path, map_location=self.device) # load the model
            self.model.load_state_dict(state_dict)
        
    # -----------------------------------
    # TRAINING LOOP
    # -----------------------------------
    def train(self): 

        # train for a given number of epochs
        createFolder(self.resultsPath)
        results = open(self.resultsPath + '/results_' + str(self.image_size) + "_"+ str(self.patch_size) +"_"+ str(self.kernel_dim) + '.txt', 'w')
        results.write('Training results: \n')
        results.write("lr: " + str(self.lr) + " batch_size: " + str(self.batchSize) + " Shape: " + str(self.image_size) + " nb_epoch " + str(self.epoch))
        results.write("Train loss\tValid Loss\n")

        best_ae = 100000
        train_losses = []
        valid_losses = []
        stats = []
        to_print = True

        for i in range(self.epoch):
            
            train_size = len(self.dataSetTrain)
            valid_size = len(self.dataSetVal)

            #############################################################################
            # Training set                                                              #
            #############################################################################
            
            train_loss = 0
            train_ae = 0

            self.model.train()
            pbar = tqdm(self.trainDataLoader)

            for _, img, bboxes_images,bboxes,_,gt,_,_ in pbar:
                
                img = img.to(self.device)
                bboxes_images = bboxes_images.to(self.device)
                bboxes = bboxes.to(self.device)
                gt = gt.to(self.device)

                self.optimizer.zero_grad()

                if to_print == True:
                    getInfoModel(self.model, img, bboxes_images,bboxes)
                    to_print = False

                out, aux_out = self.model(img, bboxes_images,bboxes)
      
                # Compute the loss
                with torch.no_grad(): num_objects = gt.sum() # count the number of objects for the loss
                main_loss = self.criterion(out, gt, num_objects)
                loss = main_loss

                if self.num_decoder_layers > 0 or self.num_ope_iterative_steps > 0:
                    aux_loss = sum([
                            self.aux_weight * self.criterion(aux, gt, num_objects) for aux in aux_out
                        ])
                    loss += aux_loss

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Update the weights
                self.optimizer.step()

                l = main_loss * img.size(0)
                train_loss += l.item()

                pred_cnt = torch.sum(out).item()
                gt_cnt = torch.sum(gt).item()
                cnt_err = abs(pred_cnt - gt_cnt)
                train_ae += cnt_err

            train_loss_mean = train_loss / train_size
            train_losses.append(train_loss_mean)
            print("mean training loss: ", str(train_loss_mean))
            
            #############################################################################
            # Valiation set                                                             #
            #############################################################################
            
            val_loss = 0
            val_ae = 0

            self.model.eval()

            pbar = tqdm(self.valDataLoader)
            with torch.no_grad():
                for _, img, bboxes_images,bboxes,_, gt,_,_ in pbar:

                    img = img.to(self.device)
                    bboxes_images = bboxes_images.to(self.device)
                    bboxes = bboxes.to(self.device)
                    gt = gt.to(self.device)

                    out, _ = self.model(img, bboxes_images,bboxes)
                    with torch.no_grad():
                        num_objects = gt.sum()

                    main_loss = (self.criterion(out, gt, num_objects))* img.size(0)
                    val_loss += main_loss.item()

                    pred_cnt = torch.sum(out).item()
                    gt_cnt   = torch.sum(gt).item()
                    cnt_err  = abs(pred_cnt - gt_cnt)
                    val_ae  += cnt_err

            valid_loss_mean = val_loss / valid_size
            valid_losses.append(valid_loss_mean)
            results.write(str(train_loss_mean) + '\t' + str(valid_loss_mean) + '\n')
        
            self.scheduler.step()
            
            stats.append((train_ae/train_size, val_ae/valid_size))
            stats_file = os.path.join(self.resultsPath, "stats" +"_"+ str(self.image_size) + ".txt")
            with open(stats_file, 'w') as f:
                for s in stats:
                    f.write("%s\n" % ','.join([str(x) for x in s]))    
            if best_ae >= val_ae/valid_size:
                print("Saving model...")
                best_ae = val_ae/valid_size
                
                model_name = self.checkpoints + '/' + self.saving_name +".pt"
                torch.save(self.model.state_dict(),model_name)

            print("Epoch {}, Train MAE: {} Val MAE: {} Best Val MAE: {}".format(
                    i,  stats[-1][0], stats[-1][1], best_ae))
                  
            
        return [value_train for value_train in train_losses], [value_valid for value_valid in valid_losses], [value[0] for value in stats], [value[1] for value in stats]




    # -------------------------------------------------
    # EVALUATION PROCEDURE
    # -------------------------------------------------
    def evaluate(self,visu=False, peaks_bool=False, save_data=False, hm=False):
        ae = torch.tensor(0.0).to(self.device) 
        se = torch.tensor(0.0).to(self.device)
        aep = 0
        sep= 0
        ground_truth = False
        self.model.eval()
        to_print = False

        start = time.time()
        pbar = tqdm(self.testDataLoader)
        cnt = 0
        with torch.no_grad():
            for im_path, img, bboxes_images, bboxes, init_boxes, gt, points, nb_points in pbar:
                img    = img.to(self.device)
                bboxes = bboxes.to(self.device)
                bboxes_images = bboxes_images.to(self.device)
                gt     = gt.to(self.device)
                
                if to_print == True:

                    def constructor(input_res):
                        return {"x":img, "references":bboxes_images, "bboxes":bboxes}
                    
                    getInfoModel(self.model, img, bboxes_images,bboxes)
                    macs, params = get_model_complexity_info(self.model, (3,self.image_size,self.image_size), input_constructor=constructor, as_strings=True, print_per_layer_stat=True, verbose=True)
                    # Extract the numerical value
                    flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
                    # Extract the unit
                    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
                    print('Computational complexity: {:<8}'.format(macs))
                    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
                    print('Number of parameters: {:<8}'.format(params))
                    to_print = False

                out,_ = self.model(img, bboxes_images,bboxes)

                # compute the error for image of the batch
                pred_cnt = torch.sum(out).item()
                gt_cnt   = torch.sum(gt).item()
                cnt_err  = abs(pred_cnt - gt_cnt)
                ae      += cnt_err
                se      += cnt_err**2
                cnt     += self.batchSize

                
                for i in range(out.shape[0]):

                    pred_cnt = round(out[i].detach().cpu().unsqueeze(0).sum().item())
                    gt_cnt   = round(gt[i].detach().cpu().unsqueeze(0).sum().item())
                    cnt_err  = abs(pred_cnt-gt_cnt)
                    relative_error = round(abs(pred_cnt-gt_cnt)/(pred_cnt+0.5),2)


                    # for the three function: size of out and gt = [h,w]
                    # self.per_pixel_comparison(out[i].squeeze(0).detach().cpu(),gt[i].squeeze(0).detach().cpu(), im_path[i],self.resultsPath)
                    min_size_object= compute_min_size_object(bboxes[i].detach().cpu().squeeze(0))
                    max_size_object= compute_max_size_object(bboxes[i].detach().cpu().squeeze(0))
                    #compute_bbr_bbmre(out[i].detach().cpu().squeeze(0),gt[i].detach().cpu().squeeze(0),points[i],self.resultsPath,im_path[i],self.image_size,self.patch_size)

                    if ground_truth:
                        resultsPath = Path(self.resultsPath)
                        resultsPath = resultsPath.parent.absolute()
                        resultsPath = os.path.join(resultsPath,"test_indu_gt")

                        if not os.path.exists(resultsPath):
                            os.mkdir(resultsPath)

                        compute_bbr_bbmre(gt[i].detach().cpu().squeeze(0),gt[i].detach().cpu().squeeze(0),points[i],resultsPath,im_path[i],self.image_size,self.patch_size)
                        #hungarian_matching(gt[i].detach().cpu().squeeze(0),points[i][:nb_points[i],:].numpy(),min_size_object, max_size_object,im_path[i],resultsPath, self.patch_size, self.kernel_dim)

                    if hm:
                        hungarian_matching(out[i].detach().cpu().squeeze(0),points[i][:nb_points[i],:].numpy(),min_size_object, max_size_object,im_path[i],self.resultsPath, self.patch_size, self.kernel_dim)
                    
                    # apply NMS
                    if peaks_bool:
                        peaks, pred_cnt_peaks, width, height = find_peaks(out[i].detach().cpu().unsqueeze(0),bboxes[i].detach().cpu(), threshold=0.15, mode='min')

                        if height > 3*width or width > 3*height: #check if we have a rectangle
                            err = np.abs(pred_cnt-gt_cnt)
                        else:
                            err = np.abs(pred_cnt_peaks-gt_cnt)

                        aep += err
                        sep += err**2

                        pbar.set_description('Current MAE: {:5.2f}, Current MAE_P: {:5.2f}'.\
                         format(ae/cnt, aep/cnt))
                    
                    else: 
                        pbar.set_description('Current MAE: {:5.2f}'.\
                         format(ae/cnt))

                    # save 3 images: 1) the input image with the bounding boxes, 2) the predicted density map, 3) both left and right
                    if visu:
                        partial_vizu(im_path[i], out[i].detach().cpu().unsqueeze(0), bboxes[i].cpu(), init_boxes[i].to('cpu'), self.resultsPath,self.model_name,gt=gt_cnt)

                    # save data statistics about the predection of the model
                    if save_data:
                        save_data_statistics(out[i],gt[i],bboxes[i],im_path[i],cnt_err, relative_error, self.resultsPath,self.image_size,self.patch_size,self.kernel_dim)


        total_time = time.time() - start
        print("Avg inf time: ", total_time/len(self.dataSetTest))              
        
        print(
            f"TEST set",
            f"MAE: {ae.item() / len(self.dataSetTest):.2f}",
            f"RMSE: {torch.sqrt(se / len(self.dataSetTest)).item():.2f}",
        )
        if peaks_bool:
            print(
                f"TEST set",
                f"MAE: {aep / len(self.dataSetTest):.2f}",
                f"RMSE: {np.sqrt(sep / len(self.dataSetTest)):.2f}",
            )

        #write result
        output_result = os.path.join(self.resultsPath, "result.txt")
        fout = open(output_result, "w")
        fout.write(str(ae/len(self.dataSetTest)) + "\n")
        fout.write(str(round(total_time,2)/len(self.dataSetTest)))
        fout.close()

        return 
    
    def get_device(self):
        return self.device
    
    def get_model(self):
        return self.model


# draft

""" if ground_truth:
    resultsPath = Path(self.resultsPath)
    resultsPath = resultsPath.parent.absolute()
    resultsPath = os.path.join(resultsPath,"test_distinct_gt")

    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)

    compute_bbr_bbmre(gt[i].detach().cpu().squeeze(0),gt[i].detach().cpu().squeeze(0),points[i],resultsPath,im_path[i],self.image_size,self.patch_size)
    hungarian_matching(gt[i].detach().cpu().squeeze(0),points[i][:nb_points[i],:].numpy(),min_size_object, max_size_object,im_path[i],resultsPath, self.patch_size, self.kernel_dim)  """