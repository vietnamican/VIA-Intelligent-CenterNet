import os
from tqdm import tqdm

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

transformer = {
    'train': T.Compose([
        T.ToPILImage(),
        T.ColorJitter(0.5, 0.5, 0.5, 0.5),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'val': T.Compose([
        T.ToPILImage(),
        T.Resize((176, 240)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    left, right = max(left, 0), min(right, width - 1)
    top, bottom = max(top, 0), min(bottom, height - 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    return round(np.sqrt(height**2 + width**2))

class TraficDataset(Dataset):
    def __init__(self, im_folder, anno_folder, mode='train'):
        super().__init__()
        self.im_names = []
        self.annos = []
        self.sigma = 2.65
        self.parse_data(im_folder, anno_folder)
        self.orig_im_height = 180
        self.im_height = 176
        self.im_width = 240
        self.transformer = transformer[mode]

    def __getitem__(self, idx):
        im_path = self.im_names[idx]
        cls, boxes = self.annos[idx][:, 0], self.annos[idx][:, 1:5]
        im = cv2.imread(im_path)
        im = im[2:-2, :]
        im = self.transformer(im)
        hm = self._make_heatmap(im, cls, boxes)
        return im, hm, im_path

    def _make_heatmap(self, im, cls, boxes):

        # 
        # 0: heatmap
        # 1, 2 : offset_x, offset_y
        # 3, 4: width, height
        scale = self.downscale
        height = self.im_height // scale
        width = self.im_width // scale

        res = np.zeros([5, height, width], dtype=np.float32)
        
        for cl, box in zip(cls, boxes):
            if cl == 0:
                break
            x_ratio, y_ratio, width_ratio, height_ratio = box

            x = x_ratio * self.im_width
            y = y_ratio * self.im_height
            width = width_ratio * self.im_width
            height = height_ratio * self.im_height

            # scaled information
            x_scaled, y_scaled = round(x / scale), round(y / scale)
            offset_x, offset_y = x / scale - x_scaled, y / scale - y_scaled
            width_scaled, height_scaled = width / scale, height / scale

            # offset
            res[1][y_scaled, x_scaled] = offset_x
            res[2][y_scaled, x_scaled] = offset_y

            # wh
            res[3][y_scaled, x_scaled] = np.log(width_scaled + 1e-4)
            res[4][y_scaled, x_scaled] = np.log(height_scaled + 1e-4)
            
            # hm
            radius = round(gaussian_radius((height_scaled, width_scaled)))
            draw_gaussian(res[0], (x_scaled, y_scaled), radius)
        return res

    def __len__(self):
        return len(self.im_names)

    def parse_data(self, im_folder, anno_folder):
        im_extension = 'jpg'
        anno_extension = 'txt'
        for im_filename in os.listdir(im_folder):
            im_name = im_filename.split('.')[0]
            full_im_filename = os.path.join(
                im_folder, im_name + '.' + im_extension)
            self.im_names.append(full_im_filename)
            full_anno_filename = os.path.join(
                anno_folder, im_name + "." + anno_extension)
            with open(full_anno_filename, 'r') as anno_file:
                lines = anno_file.read()
                lines = lines.split('\n')
                nrow = len(lines)
                if len(lines[0]) > 0:
                    annos = np.loadtxt(lines).reshape(nrow, -1)
                else:
                    annos = np.array([[0, 0, 0, 0, 0]])
                self.annos.append(annos)


if __name__ == '__main__':
    image_folder = os.path.join('via-trafficsign', 'images', 'train')
    anno_folder = os.path.join('via-trafficsign', 'labels', 'train')
    dataset = TraficDataset(image_folder, anno_folder)
    dataloader = DataLoader(dataset, batch_size=1)
    outdir = 'heatmap'
    i = 0
    for im, hm, *_ in tqdm(dataloader):
        with torch.no_grad():
            hm.squeeze_()
            hm = hm[0]
            print(type(hm))
            hm *= 255
            cv2.imwrite(os.path.join(outdir, "{}.jpg".format(i)), hm.numpy().astype(np.uint8))
        i += 1
        break
