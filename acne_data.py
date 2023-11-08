import os
import random
import numpy as np
import skimage
import skimage.transform as trans
import torch
from torch.utils.data import Dataset
from torchvision import ops
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

import utils
from config import Config


class AcneSegDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, mode: str, config: Config, transforms=None):
        super(AcneSegDataset, self).__init__()
        self.img_dir = img_dir
        self.config = config
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        self.transforms = transforms

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()

        # Anchors
        # [anchor_count, (x1, y1, x2, y2)]
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_obj = self.coco.imgs[img_id]
        ann_objs = self.coco.imgToAnns[img_id]

        image, gt_class_ids, gt_bboxes, gt_masks = \
            _load_image_gt(os.path.join(self.img_dir, img_obj['file_name']),
                           ann_objs, self.mode, self.config)

        if self.transforms is not None:
            result = self.transforms(image=image, bboxes=gt_bboxes, masks=gt_masks)
            image, gt_bboxes, gt_masks = result['image'], result['bboxes'], result['masks']

        if self.mode == 'test':
            image = torch.tensor(image.transpose((2, 0, 1)).copy()).float()  # float32 [3, H, W]
            return image, img_id

        # RPN Targets
        rpn_matches, rpn_deltas = build_rpn_targets(self.anchors, gt_bboxes, self.config)

        # If more instances than fits in the array, sub-sample from them.
        if gt_bboxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_bboxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_bboxes = gt_bboxes[ids]
            gt_masks = gt_masks[:, :, ids]
        if gt_bboxes.shape[0] < self.config.MAX_GT_INSTANCES:
            zeros = self.config.MAX_GT_INSTANCES - gt_bboxes.shape[0]
            gt_class_ids = np.concatenate(
                [gt_class_ids, np.zeros((zeros,), dtype=gt_class_ids.dtype)], axis=0)
            gt_bboxes = np.concatenate(
                [gt_bboxes, np.zeros((zeros, 4), dtype=gt_bboxes.dtype)], axis=0)
            gt_masks = np.concatenate(
                [gt_masks, np.zeros((gt_masks.shape[0], gt_masks.shape[1], zeros), dtype=gt_masks.dtype)], axis=2)

        # Convert to tensor
        image = torch.tensor(image.transpose((2, 0, 1)).copy()).float()  # float32 [3, H, W]
        rpn_matches = rpn_matches.int()  # int32 [num_anchors]
        rpn_deltas = rpn_deltas.float()  # float32 [rpn_per_image, (dx, dy, dw, dh)]
        gt_class_ids = torch.tensor(gt_class_ids).int()  # int32 [N]
        gt_bboxes = torch.tensor(gt_bboxes).float()  # float32 [N, (x1, x2, y1, y2)]
        gt_masks = torch.tensor(gt_masks.transpose((2, 0, 1)).copy()).float()  # float32 [N, m_H, m_W]

        return image, rpn_matches, rpn_deltas, gt_class_ids, gt_bboxes, gt_masks

    def __len__(self):
        return len(self.img_ids)


def _load_image_gt(img_path, ann_objs, mode, config: Config):
    # Read image
    image = skimage.io.imread(img_path)

    if mode == 'test':
        return image, None, None, None

    # Read annotation
    class_ids = np.zeros((len(ann_objs),), dtype=np.int64)
    bboxes = np.zeros((len(ann_objs), 4), dtype=np.float32)
    if config.USE_MINI_MASK:
        masks = np.zeros((config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[0], len(ann_objs)), dtype=np.float32)
    else:
        masks = np.zeros((image.shape[0], image.shape[1], len(ann_objs)), dtype=np.float32)
    for i, ann in enumerate(ann_objs):
        class_ids[i] = ann['category_id']

        bbox = ann['bbox']
        bboxes[i, :] = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

        if config.USE_MINI_MASK:
            b_w, b_h = bbox[2:]
            seg = np.array(ann['segmentation'], dtype=np.float64)
            seg[0, 0::2] -= bbox[0]
            seg[0, 1::2] -= bbox[1]
            mask = seg_to_mask(seg.tolist(), b_h, b_w)
            mask = trans.resize(mask.astype(np.float32), config.MINI_MASK_SHAPE, order=1)
        else:
            mask = seg_to_mask(ann['segmentation'], image.shape[0], image.shape[1])
        masks[:, :, i] = np.where(mask >= 0.5, 1.0, 0.0)

    if mode == 'train':
        # Random crop
        image, class_ids, bboxes, masks = random_crop(
            [image, class_ids, bboxes, masks],
            config.IMAGE_SHAPE[:2],
            config.BBOX_MIN_AREA,
            config.BBOX_MIN_HEIGHT,
            config.BBOX_MIN_WIDTH,
            config.USE_MINI_MASK)

    # Normalize bboxes
    bboxes[:, [0, 2]] /= image.shape[1]
    bboxes[:, [1, 3]] /= image.shape[0]

    return image, class_ids, bboxes, masks


def seg_to_mask(seg, height, width):
    rles = mask_utils.frPyObjects(seg, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle)


def expand_mask(bbox, mini_mask, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bbox[:4]
    h = y2 - y1
    w = x2 - x1
    m = trans.resize(mini_mask.astype(float), (h, w), order=1)
    mask[int(y1):int(y2), int(x1):int(x2)] = np.where(m >= 0.5, 1, 0)
    return mask


def random_crop(inputs, crop_size, bbox_min_area=0, bbox_min_h=0, bbox_min_w=0, use_mini_mask=True):
    image, class_ids, bboxes, masks = inputs
    shape = image.shape
    crop_h, crop_w = crop_size

    # Pad image
    if shape[0] < crop_h or shape[1] < crop_w:
        image, bboxes, masks = pad_image([image, bboxes, masks], crop_h, crop_w, use_mini_mask)
    height, width = image.shape[:2]

    x = random.randint(0, width - crop_w)
    y = random.randint(0, height - crop_h)
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(x, x + crop_w)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(y, y + crop_h)
    ws = bboxes[:, 2] - bboxes[:, 0]
    hs = bboxes[:, 3] - bboxes[:, 1]
    areas = ws * hs
    indices = np.nonzero((areas > bbox_min_area) &
                         (ws > bbox_min_w) &
                         (hs > bbox_min_h))[0]

    image = image[y:y + crop_h, x:x + crop_w, :]
    class_ids = class_ids[indices]
    bboxes = bboxes[indices, :]
    bboxes[:, [0, 2]] -= x
    bboxes[:, [1, 3]] -= y
    masks = masks[:, :, indices]

    if not use_mini_mask:
        masks = masks[y:y + crop_h, x:x + crop_w, :]

    return image, class_ids, bboxes, masks


def pad_image(inputs, min_height, min_width, ignore_mask):
    image, bboxes, masks = inputs
    height, width = image.shape[:2]
    pad_h, pad_w = 0, 0
    if height < min_height:
        pad_h = min_height - height
    if width < min_width:
        pad_w = min_width - width
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', 0)
    bboxes[:, [0, 2]] += pad_left
    bboxes[:, [1, 3]] += pad_top

    if not ignore_mask:
        masks = np.pad(masks, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', 0)

    return image, bboxes, masks


def transforms(rgb_mean, rgb_std, img_shape, mode='train'):
    assert mode in ['train', 'valid', 'test']

    def train_trans(image, bboxes, masks):
        # Random Horizontal Flip
        if random.randint(0, 1):
            image = np.fliplr(image)
            masks = np.fliplr(masks)
            bboxes[:, 0], bboxes[:, 2] = 1 - bboxes[:, 2], 1 - bboxes[:, 0]

        # Normalize Image
        image = image.astype(np.float32) / 255.0
        image = (image - rgb_mean) / rgb_std

        # Resize Image
        if image.shape[0] != img_shape[0] or image.shape[1] != img_shape[1]:
            image = trans.resize(image, img_shape, order=1)

        return {
            'image': image,
            'bboxes': bboxes,
            'masks': masks,
        }

    def infer_trans(image, bboxes, masks):
        # Normalize Image
        image = image.astype(np.float32) / 255.0
        image = (image - rgb_mean) / rgb_std

        # Resize Image
        if image.shape[0] != img_shape[0] or image.shape[1] != img_shape[1]:
            image = trans.resize(image, img_shape, order=1)

        return {
            'image': image,
            'bboxes': bboxes,
            'masks': masks,
        }

    if mode == 'train':
        return train_trans
    else:
        return infer_trans


def build_rpn_targets(anchors, gt_bboxes, config: Config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT bboxes.

    anchors: [num_anchors, (x1, y1, x2, y2)]
    gt_bboxes: [num_gt_boxes, (x1, y1, x2, y2)]

    Returns:
    rpn_matches: [num_anchors] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_deltas: [N, (dx, dy, log(dw), log(dh))] Anchor bbox deltas.
    """
    anchors = torch.tensor(anchors)
    norm = torch.tensor([config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[0],
                         config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[0]]).float()
    anchors /= norm

    gt_bboxes = torch.tensor(gt_bboxes)
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_matches = torch.zeros((anchors.size(0),))
    # RPN bounding box offsets: [max anchors per image, (dx, dy, log(dw), log(dh))]
    rpn_deltas = torch.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))
    if len(gt_bboxes) == 0:
        return rpn_matches, rpn_deltas

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = ops.box_iou(anchors, gt_bboxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them.
    anchor_iou_max, anchor_iou_argmax = torch.max(overlaps, dim=1)
    rpn_matches[anchor_iou_max < 0.3] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    gt_iou_argmax = torch.argmax(overlaps, dim=0)
    rpn_matches[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_matches[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = torch.where(rpn_matches == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = ids[torch.randperm(len(ids))][:extra]
        rpn_matches[ids] = 0
    # Same for negative proposals
    ids = torch.where(rpn_matches == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        torch.sum(rpn_matches == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = ids[torch.randperm(len(ids))][:extra]
        rpn_matches[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = torch.where(rpn_matches == 1)[0]
    rpn_deltas[:len(ids), :] = utils.box_refinement(anchors[ids], gt_bboxes[anchor_iou_argmax[ids]])
    rpn_deltas /= config.RPN_BBOX_STD_DEV

    return rpn_matches, rpn_deltas


if __name__ == '__main__':
    import time
    import tqdm
    from torch.utils.data import DataLoader

    cfg = Config()
    data_train = AcneSegDataset(os.path.join(cfg.DATA_BASE_DIR, 'images'),
                                os.path.join(cfg.DATA_BASE_DIR, 'annotations', 'acne_train.json'),
                                'train', cfg, transforms(cfg.RGB_MEAN, cfg.RGB_STD, cfg.IMAGE_SHAPE[:2], 'train'))
    data_valid = AcneSegDataset(os.path.join(cfg.DATA_BASE_DIR, 'valid_patch'),
                                os.path.join(cfg.DATA_BASE_DIR, 'annotations', 'acne_valid.json'),
                                'valid', cfg, transforms(cfg.RGB_MEAN, cfg.RGB_STD, cfg.IMAGE_SHAPE[:2], 'valid'))
    data_test = AcneSegDataset(os.path.join(cfg.DATA_BASE_DIR, 'test_patch'),
                               os.path.join(cfg.DATA_BASE_DIR, 'annotations', 'acne_test.json'),
                               'test', cfg, transforms(cfg.RGB_MEAN, cfg.RGB_STD, cfg.IMAGE_SHAPE[:2], 'test'))
    iter_train = DataLoader(data_train, batch_size=cfg.BATCH_SIZE,
                            shuffle=True, num_workers=cfg.NUM_WORKERS)
    iter_valid = DataLoader(data_valid, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS)
    iter_test = DataLoader(data_test, batch_size=cfg.BATCH_SIZE,
                           shuffle=False, num_workers=cfg.NUM_WORKERS)

    start = time.time()
    for data in tqdm.tqdm(iter_train):
        continue
    print(f'Load train data cost: {time.time() - start}sec({(time.time() - start) / len(data_train)}sec per sample)')

    start = time.time()
    for data in tqdm.tqdm(iter_valid):
        continue
    print(f'Load valid data cost: {time.time() - start}sec({(time.time() - start) / len(data_valid)}sec per sample)')

    start = time.time()
    for data in tqdm.tqdm(iter_test):
        continue
    print(f'Load test data cost: {time.time() - start}sec({(time.time() - start) / len(data_test)}sec per sample)')
