# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from .torchvision_datasets import CocoDetection as TvCocoDetection
from .torchvision_datasets import CocoDetectionFew as TvCocoDetectionFew
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1, no_cats=False, filter_pct=-1, seed=42):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.no_cats = no_cats
        if filter_pct > 0:
            num_keep = float(len(self.ids))*filter_pct
            self.ids = np.random.default_rng(seed).choice(self.ids, size=round(num_keep), replace=False).tolist()

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self.no_cats:
            target['labels'][:] = 1
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes, keep = preprocess_xywh_boxes(boxes, h, w)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def preprocess_xywh_boxes(boxes, h, w):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    return boxes, keep


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]

    no_cats = False
    if 'coco' not in args.dataset:
        no_cats = True
    filter_pct = -1
    if image_set == 'train' and args.filter_pct > 0:
        filter_pct = args.filter_pct
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), no_cats=no_cats, filter_pct=filter_pct, seed=args.seed)
    return dataset



class CocoDetectionFew(TvCocoDetectionFew):
    def __init__(self, img_folder, anns, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1, no_cats=False, filter_classes=[], shot=None, seed=42):
        super(CocoDetectionFew, self).__init__(img_folder, anns,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.no_cats = no_cats
        self.filter_classes = filter_classes
        # filter ids with anns (common set)
        for ann in anns[1:]:
            if os.path.isfile(ann) and ann.split(".")[-1] == "json":
                ids_a = set(self.ids)
                coco = COCO(ann)
                ids_b = set(coco.imgs.keys())
                self.ids = sorted(list(ids_a.intersection(ids_b)))
            elif os.path.isdir(ann):
                ids_a = set(self.ids)
                ids_b = set()
                for ann_json in os.listdir(ann):
                    if ann_json.split(".")[-1] != "json":
                        continue
                    if shot is not None:
                        if not "{}shot".format(shot) in ann_json:
                            continue
                    coco = COCO(os.path.join(ann, ann_json))
                    ids_b.update(coco.imgs.keys())
                self.ids = sorted(list(ids_a.intersection(ids_b)))
        # filter image ids
        if len(filter_classes) > 0:
            # filter image ids
            new_ids = set()
            for cat in filter_classes:
                # get image ids that contains current category
                cat_ids = self.coco.getImgIds(self.ids, cat)
                new_ids.update(cat_ids)
            self.ids = sorted(list(new_ids))

    def __getitem__(self, idx):
        # filter annotation ids with given classes
        img, target = super(CocoDetectionFew, self).__getitem__(idx, self.filter_classes)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self.no_cats:
            target['labels'][:] = 1
        return img, target


def build_fewshot(args, trainval, classset, shot):
    # trainval과 classet에 따라 사용되는 dataset:
    # train, base -> train2014.json과 trainvalno5k.json의 교집합 (=train2014.json에서 5k.json을 제외한 집합) 중 base_classes를 가지는 images, 단, 해당 images에서 novel_classes의 annotation은 제거
    # val, base -> val2014.json과 trainvalno5k.json의 교집합 (=val2014.json에서 5k.json을 제외한 집합) 중 base_classes를 가지는 images, 단, 해당 images에서 novel_classes의 annotation은 제거
    # train, all -> fewshot json들과 train2014.json의 교집합
    # val, all -> fewshot json들과 val2014.json의 교집합
    # val, novel -> 5k.json에서 novel_classes를 가지는 images, 단, 해당 images에서 base_classes의 annotation은 제거

    novel_classes = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
    base_classes = [
        8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 84, 85, 86, 87, 88, 89, 90,
    ]
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    if classset == "base":
        if trainval == "train":
            img_folder = os.path.join(str(root), "train2014")
            filter_classes = base_classes
            anns = ["/data/data/MSCoco/2014/instances_train2014.json", "/data/data/cocosplit/datasplit/trainvalno5k.json"]
        elif trainval == "val":
            img_folder = os.path.join(str(root), "val2014")
            filter_classes = base_classes
            anns = ["/data/data/MSCoco/2014/instances_val2014.json", "/data/data/cocosplit/datasplit/trainvalno5k.json"]
    elif classset =="all":
        if trainval == "train":
            img_folder = os.path.join(str(root), "train2014")
            filter_classes = []
            anns = ["/data/data/MSCoco/2014/instances_train2014.json", "/data/data/cocosplit/seed1/"]
        elif trainval == "val":
            img_folder = os.path.join(str(root), "val2014")
            filter_classes = []
            anns = ["/data/data/MSCoco/2014/instances_val2014.json", "/data/data/cocosplit/seed1/"]
    elif classset == "novel":
        assert trainval == "val", "Only validation mode is supported when novel classes is selected"
        img_folder = os.path.join(str(root), "val2014")
        filter_classes = novel_classes
        anns = ["/data/data/cocosplit/datasplit/5k.json"]

    no_cats = False
    dataset = CocoDetectionFew(img_folder, anns, transforms=make_coco_transforms(trainval), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), no_cats=no_cats, filter_classes=filter_classes, shot=shot, seed=args.seed)
    return dataset
