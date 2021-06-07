import os

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# Hard code for via-hello dataset
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


class TrafficDataset(Dataset):
    def __init__(self, im_folder, anno_folder, mode='train'):
        super().__init__()
        self.im_names = []
        self.annos = []
        self.sigma = 2.65
        self.downscale = 4
        self.parse_data(im_folder, anno_folder)

        # Hard code for via-hello dataset
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

        grid_x = np.tile(np.arange(width), reps=(height, 1))
        grid_y = np.tile(np.arange(height), reps=(width, 1)).transpose()

        for cl, box in zip(cls, boxes):
            if cl == 0:
                break
            # original information
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
            grid_dist = (grid_x - x_scaled) ** 2 + (grid_y - y_scaled) ** 2
            heatmap = np.exp(-0.5 * grid_dist / self.sigma ** 2)
            heatmap[grid_dist > width_scaled ** 2 + height_scaled ** 2] = 0
            res[0] = np.maximum(heatmap, res[0])

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
    dataset = TrafficDataset(image_folder, anno_folder)
    dataloader = DataLoader(dataset, batch_size=1)
    for im, hm in dataloader:
        with torch.no_grad():
            hm.squeeze_()
            hm[hm > 0] = 1
            np.savetxt('temp.txt', hm[0])
            break
