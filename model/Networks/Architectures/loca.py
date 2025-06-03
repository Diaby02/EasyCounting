from .backbone import ResNet, MobileNetV3, EfficientNet_b0, TinyViT
from .transformer import TransformerEncoder, TransformerDecoder
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor
from .ope import OPEModule
from .rotated_conv import batch_rotate_multiweight, RountingFunction

import torch
from torch import nn
from torch.nn import functional as F


class LOCA(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_encoder_layers: int,
        num_ope_iterative_steps: int,
        num_decoder_layers: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        train_backbone: bool,
        reduction: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        first: bool,
        use_first: bool,
        norm: bool,
        last_layer: str,
        backbone_model: str,
        device: str,
        scale_only:bool,
        scale_as_key:bool,
        trainable_references: bool,
        rotation: bool,
        trainable_rotation: bool,
        trainable_rot_nb_blocks: int
    ):

        super(LOCA, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_iterative_layers = num_ope_iterative_steps
        self.first = first
        self.use_first = use_first
        self.trainable_references = trainable_references
        self.rotation = rotation
        self.trainable_rotation = trainable_rotation
        self.device = device

        if backbone_model == "MobileNetV3":
            self.backbone = MobileNetV3(
                reduction=reduction, require_grad=train_backbone, kernel_dim=kernel_dim
            )
        elif backbone_model == "EfficientNet":
            self.backbone = EfficientNet_b0(
                reduction=reduction, require_grad=train_backbone, kernel_dim=kernel_dim
            )
        elif backbone_model == "TinyViT":
            self.backbone = TinyViT(
                reduction=reduction, require_grad=train_backbone, kernel_dim=kernel_dim
            )
        else:
            self.backbone = ResNet(
                dilation=False, reduction=reduction,
                requires_grad=train_backbone, last_layer=last_layer, kernel_dim=kernel_dim
            )
        
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, emb_dim, kernel_size=1
        )

        if num_encoder_layers > 0:
            self.encoder_query = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )
            if trainable_references:
                self.encoder_references = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                    mlp_factor, norm_first, activation, norm
                )

        if num_ope_iterative_steps > 0:
            self.ope = OPEModule(
                num_ope_iterative_steps, emb_dim, kernel_dim, num_heads,
                layer_norm_eps, mlp_factor, norm_first, activation, norm,
                scale_only,scale_as_key
            )
            self.aux_heads2 = nn.ModuleList([
                DensityMapRegressor(emb_dim, reduction)
                for _ in range(num_ope_iterative_steps - 1) # une map pour chaque iterative step
            ])

        if num_decoder_layers > 0:
            self.decoder = TransformerDecoder(num_decoder_layers,norm,emb_dim,num_heads,
                        dropout,layer_norm_eps,mlp_factor,norm_first,activation
            )

            self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction)
                for _ in range(num_decoder_layers - 1) # une map pour chaque iterative step
            ])

        if self.trainable_rotation:
            self.rounting_func = RountingFunction(in_channels=self.emb_dim, kernel_number=4, nb_depth_conv=trainable_rot_nb_blocks)

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.pos_emb = PositionalEncodingsFixed(emb_dim)

    def forward(self, x, references,bboxes):
        """ The forward expects samples containing query images and corresponding exemplar patches.
            x is a stack of query images, of shape [batch_size X 3 X H X W]
            references is a torch Tensor, of shape [batch_size x num_examples x 3 x 96 x 96]
            The size of patches are small than samples

            It returns a dict with the following elements:
               - "density_map": Shape= [batch_size x 1 X h_query X w_query]
               - "patch_feature": Features vectors for exemplars, not available during testing.
                                  They are used to compute similarity loss. 
                                Shape= [exemplar_number x bs X hidden_dim]
               - "img_feature": Feature maps for query images, not available during testing.
                                Shape= [batch_size x hidden_dim X h_query X w_query]
            
        """
        num_objects = self.num_objects

        ##############################
        # query image
        ##############################

        query_features  = self.backbone(x)
        query_features  = self.input_proj(query_features) # out: bsx256x48x48 / 64x64 if the size of the image is 512
        bs, nb_channels, h_q, w_q = query_features.size()
        pos_emb         = self.pos_emb(bs, h_q, w_q, query_features.device).flatten(2).permute(2, 0, 1)
        query_features  = query_features.flatten(2).permute(2, 0, 1) # out: bsx256x2048 then 2048xbsx256

        # push through the encoder
        if self.num_encoder_layers > 0:
            query_features = self.encoder_query(query_features, pos_emb, src_key_padding_mask=None, src_mask=None)

        #reshape 
        f_e = query_features.permute(1, 2, 0).reshape(-1, self.emb_dim, h_q, w_q) # inverse operation of flatten(2).permute(2, 0, 1) # out: bsx256x48x48

        ###############################
        # references images
        ###############################

        references = references.flatten(0, 1) 
        references_features = self.backbone(references)
        references_features = self.input_proj(references_features) # out: bsxn_objectsx256x3x3 
        bs_big,_, h_r, w_r  = references_features.size()
        pos_emb_references  = self.pos_emb(bs_big, h_r, w_r, references_features.device).flatten(2).permute(2, 0, 1)
        references_features = references_features.flatten(2).permute(2,0,1)  #out: 9x12x256

        # push through the encoder
        if self.num_encoder_layers > 0:
            if self.trainable_references:
                references_features = self.encoder_references(references_features, pos_emb_references, src_key_padding_mask=None, src_mask=None)
            else:
                references_features = self.encoder_query(references_features, pos_emb_references, src_key_padding_mask=None, src_mask=None)

        #reshape 
        f_e_references = references_features.permute(1, 0, 2).reshape(bs, num_objects, h_r, w_r, -1) # 12x9x256 then bsx3x3x3x256
        pos_emb_references = pos_emb_references.reshape(-1,bs,self.emb_dim)

        ###############################################
        # OPE Module, only if num_iterative_layers > 0
        ###############################################

        # still need to change train function in network
        # to change if we want to include rotation

        if self.num_iterative_layers:
            all_prototypes = self.ope(f_e,f_e_references,pos_emb,pos_emb_references,bboxes)
            outputs = list()
            if self.trainable_rotation:
                nb_kernel = 4
                alphas, angles = self.rounting_func(f_e) # fe : bs x emb_dim x 48 x 48

            for i in range(all_prototypes.size(0)): # number of iterative steps
                prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape( #4x27x256 puis 4x3x3x3x256
                    bs, num_objects, self.kernel_dim, self.kernel_dim, -1
                ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]          #4x3x256x3x3 puis 3072x3x3 puis 3072x1x3x3

                if self.trainable_rotation:
                    prototypes = batch_rotate_multiweight(prototypes.repeat(nb_kernel,1,1,1,1), alphas.to(self.device), angles.to(self.device),trainable=True) # 3072x1x3x3

                response_maps = F.conv2d(
                    torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0), # concat along the embedding dimension (256+256+256), then 1x(768x4)x48x48 and thus 1x3072x48x48
                    prototypes,                                                                     # 3072x1x3x3      # out_channel, in_channel/group = 1, kernel_size, kernel_size                                                      
                    bias=None,
                    padding=self.kernel_dim // 2,
                    groups=prototypes.size(0)                                                       # 3072, we will have 3072 groups, thus 1 perchannel
                ).view(                                                                             # out dim: 3072x48x48
                    bs, num_objects, self.emb_dim, h_q, w_q                                             # reshape as 4x3x256x48x48
                ).max(dim=1)[0]                                                                     # take the maximum value from the 3 objects

                # send through regression heads
                if i == all_prototypes.size(0) - 1:
                    predicted_dmaps = self.regression_head(response_maps)
                else:
                    predicted_dmaps = self.aux_heads2[i](response_maps)
                outputs.append(predicted_dmaps)

            return outputs[-1], outputs[:-1]

        ################################################
        # Decoder Module, only if num_decoder_layers > 0
        ################################################

        if self.num_decoder_layers > 0:
            #same procedure as orginal loca but not same outputs sizes
            outputs = list()
            all_responses_maps = self.decoder(f_e,f_e_references,pos_emb,pos_emb_references)

            for i in range(all_responses_maps.size(0)): #number of blocks
                # orginal size: 2048xbsx256
                response_maps = all_responses_maps[i, ...].permute(1,2,0).reshape(bs,self.emb_dim,h_q,w_q) #4x256x48x48

                # send through regression heads
                if i == all_responses_maps.size(0) - 1:
                    predicted_dmaps = self.regression_head(response_maps)
                else:
                    predicted_dmaps = self.aux_heads[i](response_maps)
                outputs.append(predicted_dmaps)

            return outputs[-1], outputs[:-1]
        
        ################################################
        # rotation
        ################################################

        if self.rotation:

            prototypes     = f_e_references.permute(0,1,4,2,3)                                               # bsxn_objectsx256x3x3
            prototypes     = prototypes.flatten(0,2)                                                         # (768xbs)x3x3 (3072 if bs=4)
            prototypes     = prototypes[:,None,...]                                                          # 3072x1x3x3
            pi             = torch.acos(torch.zeros(1)).item() * 2                                                          

            angles = [0,pi/4,pi/2,3*pi/4]
            nb_rotations = 4
            rotated_prototypes = torch.cat([batch_rotate_multiweight(prototypes.unsqueeze(0),torch.ones(1,1).to(self.device),torch.Tensor([angles[i]]).to(self.device).unsqueeze(0)) for i in range(nb_rotations)]) #3072*nb_rotationsx1x3x3 # batch size = 1 and number of angle = 1 to not change the function
            # weights: 1x3072x1x3x3 | thetas: 1x1 | alphas: 1x1 | weights_out: 3072x1x3x3
            
            #depthwise correlation
            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects*nb_rotations)], dim=1).flatten(0, 1).unsqueeze(0), # concat along the embedding dimension (256+256+256), then (768x4)x48x48 and thus 1x3072x48x48
                rotated_prototypes,                                                                     # 3072x1x3x3      # out_channel, in_channel/group = 1, kernel_size, kernel_size                                                      
                bias=None,
                padding=self.kernel_dim // 2,
                groups=rotated_prototypes.size(0)                                                       # 3072, we will have 3072 groups, thus 1 perchannel
            ).view(                                                                             # out dim: 3072x48x48
                bs, num_objects*nb_rotations, self.emb_dim, h_q, w_q                                         # reshape as 4x3x256x48x48
            ).max(dim=1)[0]                                                                     # take the maximum value from the 3 objects

            # send through regression heads
            predicted_dmaps = self.regression_head(response_maps)

        ################################################
        # trainable rotation
        ################################################
        
        if self.trainable_rotation:
            prototypes     = f_e_references.permute(0,1,4,2,3)                                               # bsxn_objectsx256x3x3
            prototypes     = prototypes.flatten(0,2)                                                         # (768xbs)x3x3 (3072 if bs=4)
            prototypes     = prototypes[:,None,...]                                                          # 3072x1x3x3

            nb_kernel = 4
            alphas, angles = self.rounting_func(f_e) # fe : bs x emb_dim x 48 x 48

            rotated_prototypes = batch_rotate_multiweight(prototypes.repeat(nb_kernel,1,1,1,1), alphas.to(self.device), angles.to(self.device),trainable=True) # 3072x1x3x3

            #depthwise correlation
            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0), # concat along the embedding dimension (256+256+256), then 1x(768x4)x48x48 and thus 1x3072x48x48
                rotated_prototypes,                                                                     # 3072x1x3x3      # out_channel, in_channel/group = 1, kernel_size, kernel_size                                                      
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)                                                       # 3072, we will have 3072 groups, thus 1 perchannel
            ).view(                                                                             # out dim: 3072x48x48
                bs, num_objects, self.emb_dim, h_q, w_q                                         # reshape as 4x3x256x48x48
            ).max(dim=1)[0]                                                                     # take the maximum value from the 3 objects

            # send through regression heads
            predicted_dmaps = self.regression_head(response_maps)




        ################################################
        # Classical Depth-wise correlation
        ################################################

        else:
            prototypes     = f_e_references.permute(0,1,4,2,3)                                               # bsxn_objectsx256x3x3
            prototypes     = prototypes.flatten(0,2)                                                         # (768xbs)x3x3 (3072 if bs=4)
            prototypes     = prototypes[:,None,...]                                                          # 3072x1x3x3

            #depthwise correlation
            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0), # concat along the embedding dimension (256+256+256), then 1x(768x4)x48x48 and thus 1x3072x48x48
                prototypes,                                                                     # 3072x1x3x3      # out_channel, in_channel/group = 1, kernel_size, kernel_size                                                      
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)                                                       # 3072, we will have 3072 groups, thus 1 perchannel
            ).view(                                                                             # out dim: 3072x48x48
                bs, num_objects, self.emb_dim, h_q, w_q                                         # reshape as 4x3x256x48x48
            ).max(dim=1)[0]                                                                     # take the maximum value from the 3 objects

            # send through regression heads
            predicted_dmaps = self.regression_head(response_maps)

        return predicted_dmaps, None


def build_model(param):

    return LOCA(
        image_size= param["DATASET"]["IMAGE_SIZE"],
        num_encoder_layers= param["MODEL"]["NUM_ENC_LAYERS"],
        num_decoder_layers= param["MODEL"]["NUM_DECODER_LAYERS"],
        num_ope_iterative_steps= param["MODEL"]["NUM_OPE_ITERATIVE_STEPS"],
        num_objects= param["MODEL"]["NUM_OBJECTS"],
        emb_dim= param["MODEL"]["EMB_DIM"],
        num_heads= param["MODEL"]["NUM_HEADS"],
        kernel_dim= param["MODEL"]["KERNEL_DIM"],
        train_backbone= param["TRAINING"]["BACKBONE_LR"] > 0,
        reduction= param["MODEL"]["REDUCTION"],
        dropout= param["TRAINING"]["DROPOUT"],
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first= param["TRAINING"]["PRENORM"],
        activation=nn.GELU,
        first= param['TRAINING']['FIRST'],
        use_first= param['TRAINING']['USE_FIRST'],
        last_layer= param['MODEL']['LAST_LAYER'],
        backbone_model= param['MODEL']['BACKBONE_MODEL'],
        device= param['TRAINING']['DEVICE'],
        scale_only=param["MODEL"]["SCALE_ONLY"],
        scale_as_key=param["MODEL"]["SCALE_AS_KEY"],
        trainable_references=param["MODEL"]["TRAINABLE_REFERENCES"],
        rotation=param["MODEL"]["ROTATION"],
        trainable_rotation=param["MODEL"]["TRAINABLE_ROTATION"],
        trainable_rot_nb_blocks=param["MODEL"]["TRAINABLE_ROT_NB_BLOCKS"],
        norm=True,
    )
