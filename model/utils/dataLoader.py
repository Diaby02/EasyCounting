import os
import json
import argparse

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision import transforms as T
from torchvision.transforms import functional as TVF
from utils.utils_ import expand2square

from tqdm import tqdm


IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def tiling_augmentation(img, bboxes, density_map, points, resize, jitter, tile_size, hflip_p):

    def apply_hflip(tensor, apply):
        return TVF.hflip(tensor) if apply else tensor

    def make_tile(x, num_tiles, hflip, hflip_p, jitter=None):
        result = list()
        for j in range(num_tiles):
            row = list()
            for k in range(num_tiles):
                t = jitter(x) if jitter is not None else x
                if hflip[j, k] < hflip_p:
                    t = TVF.hflip(t)
                row.append(t)
            result.append(torch.cat(row, dim=-1))
        return torch.cat(result, dim=-2)

    x_tile, y_tile = tile_size
    y_target, x_target = resize.size
    num_tiles = max(int(x_tile.ceil()), int(y_tile.ceil()))
    # whether to horizontally flip each tile
    hflip = torch.rand(num_tiles, num_tiles)

    img = make_tile(img, num_tiles, hflip, hflip_p, jitter)
    img = resize(img[..., :int(y_tile*y_target), :int(x_tile*x_target)])

    density_map = make_tile(density_map, num_tiles, hflip, hflip_p)
    density_map = density_map[..., :int(y_tile*y_target), :int(x_tile*x_target)]
    original_sum = density_map.sum()
    density_map = resize(density_map)
    density_map = density_map / density_map.sum() * original_sum

    if hflip[0, 0] < hflip_p:
        bboxes[:, [0, 2]] = x_target - bboxes[:, [2, 0]]  # TODO change
        points[:, [0]]    = x_target - points[:, [0]]
        
    bboxes = bboxes / torch.tensor([x_tile, y_tile, x_tile, y_tile])
    points = points / torch.tensor([x_tile,y_tile])
    return img, bboxes, density_map, points


class FSCDataset(Dataset):

    def __init__(
        self, data_path, img_size,patch_size, image_folder, gt_folder, split_file, annotation_file, split='train', num_objects=3,
        tiling_p=0.5, zero_shot=False, padding=False, patch_size_ratio = 1
    ):
        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_size_ratio = patch_size_ratio
        self.resize = T.Resize((img_size, img_size))
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        self.padding = padding

        #brand new attributes
        self.image_folder      = image_folder
        self.gt_folder         = gt_folder
        self.annotation_folder = annotation_file
        #change here 
        with open(
            os.path.join(self.data_path, split_file), 'rb'
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(
            os.path.join(self.data_path, annotation_file), 'rb'
        ) as file:
            self.annotations = json.load(file)

    def __getitem__(self, idx: int):
        im_path = os.path.join(
            self.data_path,
            self.image_folder,
            self.image_names[idx]
        )
        img = Image.open(im_path).convert("RGB")
        w, h = img.size
        if self.split != 'train':
            img = T.Compose([
                T.ToTensor(),
                self.resize,
                T.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
            ])(img)
        else:
            img = T.Compose([
                T.ToTensor(),
                self.resize,
            ])(img)

        #Coordinates version of the bboxes
        init_bboxes = torch.tensor(
            self.annotations[self.image_names[idx]]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...] #[x1, y1, x2, y2]
        bboxes = (init_bboxes / torch.tensor([w,h,w,h])) * self.img_size

        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            self.gt_folder,
            os.path.splitext(self.image_names[idx])[0] + '.npy',
        ))).unsqueeze(0)

        #Coordinates version of the bboxes
        init_points = torch.tensor(
            self.annotations[self.image_names[idx]]['points'],
            dtype=torch.float32
        ) #[x1, y1, x2, y2]
        points = (init_points / torch.tensor([w,h])) * self.img_size
        
        original_sum = density_map.sum()
        density_map = self.resize(density_map)
        density_map = density_map / density_map.sum() * original_sum

        # data augmentation
        tiled = False
        if self.split == 'train' and torch.rand(1) < self.tiling_p:
            tiled = True
            tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
            img, bboxes, density_map , points = tiling_augmentation(
                img, bboxes, density_map, points, self.resize,
                self.jitter, tile_size, self.horizontal_flip_p
            )

        if self.split == 'train':
            if not tiled:
                img = self.jitter(img)
            img = T.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)(img)

        if self.split == 'train' and not tiled and torch.rand(1) < self.horizontal_flip_p:
            img = TVF.hflip(img)
            density_map = TVF.hflip(density_map)
            bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]
            points[:, [0]]    = self.img_size - points[:, [0]]

        # Image version of the bboxes
        box_trans = T.Compose([
                T.Resize((int(self.patch_size), int(self.patch_size* self.patch_size_ratio))),
                T.Normalize(mean=IM_NORM_MEAN,
                                    std=IM_NORM_STD)
            ])
        
        box_trans2 = T.Compose([
                T.ToTensor(),
                T.Resize((int(self.patch_size), int(self.patch_size))),
                T.Normalize(mean=IM_NORM_MEAN,
                                    std=IM_NORM_STD)
            ])
        
        bboxes_images = []
        for i in range(0, bboxes.shape[0]):
            x1, y1, x2, y2 = int(bboxes[i, 0].item()), int(bboxes[i, 1].item()), int(bboxes[i, 2].item()), int(
                bboxes[i, 3].item())
            box_ = img[:,y1:y2, x1:x2]

            if self.padding:
                box_ = expand2square(TVF.to_pil_image(box_), (0,0,0))
                bboxes_images.append(box_trans2(box_))
            else:
                bboxes_images.append(box_trans(box_))
        bboxes_images = torch.stack(bboxes_images, dim=0)

        target = torch.zeros(4000,2)
        nb_points = points.shape[0]
        target[:nb_points,:] = points
        points = target

        return im_path, img, bboxes_images, bboxes, init_bboxes, density_map, points, nb_points

    def __len__(self):
        return len(self.image_names)
    
def resize_img(img, bboxes, img_size, num_objects):
    w, h = img.size

    if bboxes is not None:

        # Image version of the bboxes
        box_trans = T.Compose([
                T.Resize((36,36)),
                T.ToTensor(),
                T.Normalize(mean=IM_NORM_MEAN,
                                    std=IM_NORM_STD)
            ])
        
        bboxes_images = []
        for box in bboxes:
            box_ = img.crop(tuple(np.array(box)))
            bboxes_images.append(box_trans(box_))
        bboxes_images = torch.stack(bboxes_images, dim=0)

        # Coordinates version of the bboxes
        bboxes = torch.tensor(
            [bboxes],
            dtype=torch.float32
        )[:num_objects, ...]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * img_size

    img = T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size)),
        T.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
    ])(img)

    return img, bboxes, bboxes_images

# work only with FSC147 dataset
def generate_density_maps(data_path, target_size=(512, 512)):

    density_map_path = os.path.join(
        data_path,
        f'gt_density_map_adaptive_{target_size[0]}_{target_size[1]}_object_VarV2'
    )
    if not os.path.isdir(density_map_path):
        os.makedirs(density_map_path)

    with open(
        os.path.join(data_path, 'annotation_FSC147_384.json'), 'rb'
    ) as file:
        annotations = json.load(file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for i, (image_name, ann) in enumerate(tqdm(annotations.items())):
        _, h, w = T.ToTensor()(Image.open(os.path.join(
            data_path,
            'images_384_VarV2',
            image_name
        ))).size()
        h_ratio, w_ratio = target_size[0] / h, target_size[1] / w

        points = (
            torch.tensor(ann['points'], device=device) *
            torch.tensor([w_ratio, h_ratio], device=device)
        ).long()
        points[:, 0] = points[:, 0].clip(0, target_size[1] - 1)
        points[:, 1] = points[:, 1].clip(0, target_size[0] - 1)
        bboxes = box_convert(torch.tensor(
            ann['box_examples_coordinates'],
            dtype=torch.float32,
            device=device
        )[:3, [0, 2], :].reshape(-1, 4), in_fmt='xyxy', out_fmt='xywh')
        bboxes = bboxes * torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio], device=device)
        window_size = bboxes.mean(dim=0)[2:].cpu().numpy()[::-1]

        dmap = torch.zeros(*target_size)
        for p in range(points.size(0)):
            dmap[points[p, 1], points[p, 0]] += 1
        dmap = gaussian_filter(dmap.cpu().numpy(), window_size / 8)

        np.save(os.path.join(density_map_path, os.path.splitext(image_name)[0] + '.npy'), dmap)


if __name__ == '__main__':
    # python utils/data.py --data_path <path_to_your_data_directory> --image_size 512 
    parser = argparse.ArgumentParser("Density map generator", add_help=False)
    parser.add_argument(
        '--data_path',
        default='/home/nibou/Documents/Master_thesis_Euresys/loca/data/',
        type=str
    )
    parser.add_argument('--image_size', default=512, type=int)
    args = parser.parse_args()
    generate_density_maps(args.data_path, (args.image_size, args.image_size))
