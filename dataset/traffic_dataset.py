import os
import os.path
import torch
import torch.utils.data as data
import cv2
import numpy as np
import albumentations as A
# Declare an augmentation pipeline
category_ids = [0]
transformerr = A.Compose(
    [
        # A.HorizontalFlip(p=0.5),  ## Becareful when using that, because the keypoint is flipped but the index is flipped too
        A.ColorJitter(brightness=0.35, contrast=0.5,
                      saturation=0.5, hue=0.2, always_apply=False, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.25, rotate_limit=30,
                           interpolation=1, border_mode=4, always_apply=False, p=1)

    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])
)


def augment(img, boxes):
    out = transformerr(image=img, bboxes=boxes)
    return out['image'], out['bboxes']
# Usage
# transformed = transformerr(image=img, bboxes=[box], category_ids=category_ids,keypoints=landmark )
# imgT = np.array(transformed["image"])
# boxes = np.array(transformed["bboxes"])
# lmks = np.array(transformed["keypoints"])


def yolobb2pascalbb(bboxes, size):
    h, w = size
    
    bboxes[:, 1] -= bboxes[:, 3] / 2
    bboxes[:, 2] -= bboxes[:, 4] / 2
    bboxes[:, 3] += bboxes[:, 1]
    bboxes[:, 4] += bboxes[:, 2]
    bboxes[:, (1, 3)] *= w
    bboxes[:, (2, 4)] *= h
    bboxes[:, 1:] -= 1

    return bboxes


class TrafficDataset(data.Dataset):
    def __init__(self, img_dir, label_dir, preproc=None, augment=False):
        self.preproc = preproc
        self.augment = augment
        self.img_paths = []
        self.labels = []
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.__parse_file()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        height, width, _ = img.shape

        labels = self.labels[index]
        if self.augment:
            try:
                img, labels = augment(img, labels)
                labels = np.array(labels)
            except:
                pass
        labels = yolobb2pascalbb(labels, (height, width))
        annotations = np.zeros((0, 15))
        annotation = np.zeros((1, 15))
        # bbox
        for label in labels:
            annotation[0, 0] = label[1]  # x1
            annotation[0, 1] = label[2]  # y1
            annotation[0, 2] = label[1] + label[3]  # x2
            annotation[0, 3] = label[2] + label[4]  # y2

            annotation[0, 4] = -1
            annotation[0, 5] = -1
            annotation[0, 6] = -1
            annotation[0, 7] = -1
            annotation[0, 8] = -1
            annotation[0, 9] = -1
            annotation[0, 10] = -1
            annotation[0, 11] = -1
            annotation[0, 12] = -1
            annotation[0, 13] = -1
            annotation[0, 14] = label[0]
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)

        if self.preproc is not None:
            img, target = self.preproc(img, target)
        return torch.from_numpy(img), target

    def __parse_file(self):
        label_extension = 'txt'
        for img_file in os.listdir(self.img_dir):
            img_name, img_extension = os.path.splitext(img_file)
            self.img_paths.append(os.path.join(self.img_dir, img_file))
            label_file_path = os.path.join(
                self.label_dir, img_name + "." + label_extension)
            with open(label_file_path, 'r') as f:
                content = f.read().split('\n')
                label = np.loadtxt(content).reshape(-1, 5)
                # label = label[1:]
                self.labels.append(label)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


if __name__ == '__main__':
    img_dim = 96
    rgb_mean = (104, 117, 123)
    dataset = LaPa(
        os.path.join('data', 'train', 'images'),
        os.path.join('data', 'train', 'labels'),
        # preproc(img_dim, rgb_mean)
    )
    dataloader = data.DataLoader(dataset, 2, shuffle=True, num_workers=1)
    for img, target in dataloader:
        print(target.shape)
