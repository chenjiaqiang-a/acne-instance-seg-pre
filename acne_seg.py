import os
import time
import argparse
import logging

import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval

import utils
from config import Config
from model import MaskRCNN
from acne_data import AcneSegDataset, transforms, expand_mask

###################################################################
# Global Variables
###################################################################
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save running information
RUNNING_INFO_DIR = os.path.join(ROOT_DIR, 'run')
# Directory to save tensorboard logs
TENSORBOARD_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Directory path to save model checkpoints
MODAL_SAVE_DIR = os.path.join(ROOT_DIR, 'checkpoints')

# Path to trained weights file
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "mask_rcnn_acne.pth")

# Default device
_device = torch.device('cpu')

###################################################################
# Command Line Arguments
###################################################################
parser = argparse.ArgumentParser(description='Train Mask R-CNN on ACNE')
parser.add_argument('command', metavar='<command>',
                    help='"train" or "test" on ACNE')
parser.add_argument('--running_info', required=False,
                    default=RUNNING_INFO_DIR,
                    metavar='/path/to/save/running/info',
                    help='Running info directory (default=run/)')
parser.add_argument('--tensorboard_logs', required=False,
                    default=TENSORBOARD_LOGS_DIR,
                    metavar="/path/to/tensorboard/logs/",
                    help='Tensorboard logs directory (default=logs/)')
parser.add_argument('--model_save', required=False,
                    default=MODAL_SAVE_DIR,
                    metavar='/path/to/save/model/checkpoint/',
                    help='Model saving directory (default=checkpoints/')
parser.add_argument('--checkpoint', required=False,
                    metavar="/path/to/weights.pth",
                    help="Path to weights .pth file or 'default'")
args = parser.parse_args()
RUNNING_INFO_DIR = args.running_info
TENSORBOARD_LOGS_DIR = args.tensorboard_logs
MODAL_SAVE_DIR = args.model_save
if not os.path.exists(RUNNING_INFO_DIR):
    os.makedirs(RUNNING_INFO_DIR)
if not os.path.exists(TENSORBOARD_LOGS_DIR):
    os.makedirs(TENSORBOARD_LOGS_DIR)
if not os.path.exists(MODAL_SAVE_DIR):
    os.makedirs(MODAL_SAVE_DIR)

# Create logger
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
start_time = time.strftime('%y-%m-%d-%H%M', time.localtime(time.time()))
# file handler
fh = logging.FileHandler(os.path.join(RUNNING_INFO_DIR, start_time+'.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
_logger.addHandler(fh)

# Create SummaryWriter
_writer = SummaryWriter(TENSORBOARD_LOGS_DIR)

###################################################################
# Train
###################################################################

def train(model, train_loader, valid_loader, layers, learning_rate, config):
    """Train the model.
    model: The Mask R-CNN model.
    train_loader, valid_loader: Training and validation Dataloader objects.
    layers: Allows selecting wich layers to train. It can be:
        - A regular expression to match layer names to train
        - One of these predefined values:
            heaads: The RPN, classifier and mask heads of the network
            all: All the layers
            3+: Train Resnet stage 3 and up
            4+: Train Resnet stage 4 and up
            5+: Train Resnet stage 5 and up
    learning_rate: The learning rate to train with
    epochs: Number of training epochs. Note that previous training epochs
            are considered to be done alreay, so this actually determines
            the epochs to train in total rather than in this particaular
            call.
    """
    # Pre-defined layer regular expressions
    layer_regex = {
        # all layers but the backbone
        "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        # From a specific Resnet stage and up
        "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        # All layers
        "all": ".*",
    }
    train_epochs = {
        'heads': 4000,
        '3+': 12000,
        '4+': 12000,
        '5+': 12000,
        'all': 16000.
    }
    assert layers in layer_regex.keys()
    utils.set_trainable(model, layer_regex[layers])
    epochs = train_epochs[layers]

    # Optimizer object
    # Add L2 Regularization
    # Skip gamma and beta weights of batch normalization layers.
    trainables_wo_bn = [param for name, param in model.named_parameters(
    ) if param.requires_grad and 'bn' not in name]
    trainables_only_bn = [param for name, param in model.named_parameters(
    ) if param.requires_grad and 'bn' in name]
    optimizer = optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': config.WEIGHT_DECAY},
        {'params': trainables_only_bn}
    ], lr=learning_rate, momentum=config.LEARNING_MOMENTUM)

    train_iter = iter(train_loader)
    loss_sum = 0
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        try:
            inputs = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs = next(train_iter)
        
        # Training
        loss, loss_rpn_class, loss_rpn_bbox,\
            loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask = \
            train_epoch(model, inputs, optimizer)
        
        _writer.add_scalar(f'{layers}/loss', loss, epoch)
        _writer.add_scalar(f'{layers}/loss_rpn_class', loss_rpn_class, epoch)
        _writer.add_scalar(f'{layers}/loss_rpn_bbox', loss_rpn_bbox, epoch)
        _writer.add_scalar(f'{layers}/loss_mrcnn_class', loss_mrcnn_class, epoch)
        _writer.add_scalar(f'{layers}/loss_mrcnn_bbox', loss_mrcnn_bbox, epoch)
        _writer.add_scalar(f'{layers}/loss_mrcnn_mask', loss_mrcnn_mask, epoch)

        loss_sum += loss.detach().cpu().item() / config.VALID_INTERVAL
        # Validation
        if epoch % config.VALID_INTERVAL == 0:
            val_loss, val_loss_rpn_class, val_loss_rpn_bbox,\
                val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask = \
                valid_epoch(model, valid_loader)

            _writer.add_scalar(f'{layers}/val_loss', val_loss, epoch)
            _writer.add_scalar(f'{layers}/val_loss_rpn_class', val_loss_rpn_class, epoch)
            _writer.add_scalar(f'{layers}/val_loss_rpn_bbox', val_loss_rpn_bbox, epoch)
            _writer.add_scalar(f'{layers}/val_loss_mrcnn_class', val_loss_mrcnn_class, epoch)
            _writer.add_scalar(f'{layers}/val_loss_mrcnn_bbox', val_loss_mrcnn_bbox, epoch)
            _writer.add_scalar(f'{layers}/val_loss_mrcnn_mask', val_loss_mrcnn_mask, epoch)

            _logger.info(f'epoch {epoch:04d} | train loss {loss:.6f} | valid loss {val_loss:.6f}')

        # Save model
        if epoch % config.SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(
                MODAL_SAVE_DIR, f'{config.BACKBONE_ARCH}_{layers}_epoch_{epoch:06d}.pth'))


def train_epoch(model, inputs, optimizer):
    model.train()

    images = inputs[0].to(_device)  # [BS, 3, H, W]
    rpn_matches = inputs[1].to(_device)  # [BS, num_anchors, 1]
    rpn_deltas = inputs[2].to(_device)  # [BS, rpn_per_image, 4]
    gt_class_ids = inputs[3].to(_device)  # [BS, N]
    gt_bboxes = inputs[4].to(_device)  # [BS, N, 4]
    gt_masks = inputs[5].to(_device)  # [BS, N, m_H, m_W]

    # Run object detection
    rpn_pred_logits, rpn_pred_deltas,\
        target_class_ids, mrcnn_class_logits,\
        target_deltas, mrcnn_deltas,\
        target_masks, mrcnn_masks = \
        model([images, gt_class_ids, gt_bboxes, gt_masks])

    # Compute losses
    rpn_class_loss = compute_rpn_class_loss(rpn_pred_logits, rpn_matches)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_pred_deltas, rpn_deltas, rpn_matches)
    mrcnn_class_loss = compute_mrcnn_class_loss(mrcnn_class_logits, target_class_ids)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(mrcnn_deltas, target_deltas, target_class_ids)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(mrcnn_masks, target_masks, target_class_ids)
    loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    return loss, rpn_class_loss, rpn_bbox_loss,\
        mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss


@torch.no_grad()
def valid_epoch(model, dataloader):
    loss_sum = 0
    loss_rpn_class_sum = 0
    loss_rpn_bbox_sum = 0
    loss_mrcnn_class_sum = 0
    loss_mrcnn_bbox_sum = 0
    loss_mrcnn_mask_sum = 0

    model.train()
    steps = len(dataloader)
    for inputs in tqdm.tqdm(dataloader):
        images = inputs[0].to(_device)
        rpn_matches = inputs[1].to(_device)
        rpn_deltas = inputs[2].to(_device)
        gt_class_ids = inputs[3].to(_device)
        gt_boxes = inputs[4].to(_device)
        gt_masks = inputs[5].to(_device)

        # Run object detection
        rpn_class_logits, rpn_pred_deltas,\
            target_class_ids, mrcnn_class_logits,\
            target_deltas, mrcnn_deltas,\
            target_masks, mrcnn_masks = \
            model([images, gt_class_ids, gt_boxes, gt_masks])

        # Compute losses
        rpn_class_loss = compute_rpn_class_loss(rpn_class_logits, rpn_matches)
        rpn_bbox_loss = compute_rpn_bbox_loss(rpn_pred_deltas, rpn_deltas, rpn_matches)
        mrcnn_class_loss = compute_mrcnn_class_loss(mrcnn_class_logits, target_class_ids)
        mrcnn_bbox_loss = compute_mrcnn_bbox_loss(mrcnn_deltas, target_deltas, target_class_ids)
        mrcnn_mask_loss = compute_mrcnn_mask_loss(mrcnn_masks, target_masks, target_class_ids)
        loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

        # Statistics
        loss_sum += loss.cpu().item() / steps
        loss_rpn_class_sum += rpn_class_loss.cpu().item() / steps
        loss_rpn_bbox_sum += rpn_bbox_loss.cpu().item() / steps
        loss_mrcnn_class_sum += mrcnn_class_loss.cpu().item() / steps
        loss_mrcnn_bbox_sum += mrcnn_bbox_loss.cpu().item() / steps
        loss_mrcnn_mask_sum += mrcnn_mask_loss.cpu().item() / steps

    return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum,\
        loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum


def compute_rpn_class_loss(rpn_class_logits, rpn_matches):
    """RPN anchor classifier loss.

    rpn_matches: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_classes = (rpn_matches == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_matches != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices[:, 0], indices[:, 1], :]
    anchor_classes = anchor_classes[indices[:, 0], indices[:, 1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_classes)

    return loss


def compute_rpn_bbox_loss(rpn_pred_deltas, rpn_target_deltas, rpn_matches):
    """Return the RPN bounding box loss graph.

    rpn_pred_deltas: [batch, max positive anchors, (dx, dy, log(dw), log(dh))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_matches: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_target_deltas: [batch, anchors, (dx, dy, log(dw), log(dh))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_matches == 1)

    # Pick bbox deltas that contribute to the loss
    preds = rpn_pred_deltas[indices[:, 0], indices[:, 1], :]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_count = torch.sum((rpn_matches == 1).long(), dim=1)
    ind = torch.cat([torch.arange(x) for x in batch_count])
    targets = rpn_target_deltas[indices[:, 0], ind, :]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(preds, targets)

    return loss


def compute_mrcnn_class_loss(pred_class_logits, target_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    target_class_ids = target_class_ids.reshape((-1,)).long()
    pred_class_logits = pred_class_logits.reshape((-1, pred_class_logits.size(-1)))

    loss = F.cross_entropy(pred_class_logits, target_class_ids)
    return loss


def compute_mrcnn_bbox_loss(pred_deltas, target_deltas, target_class_ids):
    """Loss for Mask R-CNN bounding box refinement.

    target_deltas: [batch, num_rois, (dx, dy, log(dw), log(dh))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_deltas: [batch, num_rois, num_classes, (dx, dy, log(dw), log(dh))]
    """

    loss = torch.tensor(0).float().to(_device)
    if torch.nonzero(target_class_ids > 0).size(0):
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)
        positive_roi_class_ids = target_class_ids[positive_roi_ix[:, 0], positive_roi_ix[:, 1]].long()
        indices = torch.cat((positive_roi_ix, positive_roi_class_ids.unsqueeze(1)), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        targets = target_deltas[indices[:, 0], indices[:, 1], :]
        preds = pred_deltas[indices[:, 0], indices[:, 1], indices[:, 2], :]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(preds, targets)

    return loss


def compute_mrcnn_mask_loss(pred_masks, target_masks, target_class_ids):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, num_classes, height, width] float32 tensor
                with values from 0 to 1.
    """
    loss = torch.tensor(0).float().to(_device)
    if torch.nonzero(target_class_ids > 0).size(0):
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)
        positive_class_ids = target_class_ids[positive_ix[:, 0], positive_ix[:, 1]].long()
        indices = torch.cat((positive_ix, positive_class_ids.unsqueeze(1)), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:, 0], indices[:, 1], :, :]
        y_pred = pred_masks[indices[:, 0], indices[:, 1], indices[:, 2], :, :]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)

    return loss

###################################################################
# Evaluate
###################################################################

def evaluate(model, dataset, config):
    total_time = 0
    dataloader = DataLoader(
        dataset, config.BATCH_SIZE * 2, shuffle=False, num_workers=config.NUM_WORKERS)

    pred_results = []
    for images, img_ids in tqdm.tqdm(dataloader):
        start = time.time()
        pred_class_ids, pred_scores, pred_bboxes, pred_masks = infer(model, images, config)
        total_time += time.time() - start

        for i in range(len(img_ids)):
            pred_results.extend(generate_coco_format_result(
                img_ids[i].item(),
                pred_class_ids[i],
                pred_scores[i],
                pred_bboxes[i],
                pred_masks[i],
                config.IMAGE_SHAPE[:2],
            ))

    _logger.info(f'Total Time: {total_time} sec({total_time / len(dataset)}sec/image)')

    pred_results = dataset.coco.loadRes(pred_results)

    eval_bbox = COCOeval(dataset.coco, pred_results, 'bbox')
    eval_bbox.evaluate()
    eval_bbox.accumulate()
    eval_bbox.summarize()
    _logger.info(f'\nEvaluate bbox\n'
                 f'AP   = {eval_bbox.stats[0]:.4f}\n'
                 f'AP50 = {eval_bbox.stats[1]:.4f}\n'
                 f'AP75 = {eval_bbox.stats[2]:.4f}')

    eval_mask = COCOeval(dataset.coco, pred_results, 'segm')
    eval_mask.evaluate()
    eval_mask.accumulate()
    eval_mask.summarize()
    _logger.info(f'\nEvaluate segm\n'
                 f'AP   = {eval_mask.stats[0]:.4f}\n'
                 f'AP50 = {eval_mask.stats[1]:.4f}\n'
                 f'AP75 = {eval_mask.stats[2]:.4f}')


@torch.no_grad()
def infer(model, images, config):
    model.eval()
    images = images.to(_device)

    bs = images.size(0)
    detections, masks = model([images])
    bboxes, class_ids, scores = detections[:, :, :4], detections[:, :, 4].long(), detections[:, :, 5]

    pred_class_ids = []
    pred_scores = []
    pred_bboxes = []
    pred_masks = []
    for i in range(bs):
        # Filter out background
        idx = torch.nonzero(class_ids[i])[:, 0]
        b_class_ids = class_ids[i, idx]
        b_scores = scores[i, idx]
        b_bboxes = bboxes[i, idx]
        b_masks = masks[i, idx, b_class_ids]

        b_bboxes[:, [0, 2]] *= config.IMAGE_SHAPE[0]
        b_bboxes[:, [1, 3]] *= config.IMAGE_SHAPE[1]
        b_bboxes = utils.clip_boxes(b_bboxes, [0, 0, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]])
        b_bboxes = b_bboxes.round()

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        areas = (b_bboxes[:, 2] - b_bboxes[:, 0]) * (b_bboxes[:, 3] - b_bboxes[:, 1])
        idx = torch.nonzero(areas > 0)[:, 0]
        b_class_ids = b_class_ids[idx]
        b_scores = b_scores[idx]
        b_bboxes = b_bboxes[idx]
        b_masks = b_masks[idx]

        pred_class_ids.append(b_class_ids.int().cpu().numpy())
        pred_scores.append(b_scores.cpu().numpy())
        pred_bboxes.append(b_bboxes.cpu().numpy())
        pred_masks.append(b_masks.cpu().numpy().transpose((1, 2, 0)))

    return pred_class_ids, pred_scores, pred_bboxes, pred_masks


def generate_coco_format_result(image_id, class_ids, scores, bboxes, masks, shape):
    results = []
    for i in range(len(class_ids)):
        mask = expand_mask(bboxes[i], masks[:, :, i], shape)
        results.append({
            'image_id': image_id,
            'category_id': class_ids[i],
            'score': scores[i],
            'bbox': [bboxes[i][0], bboxes[i][1], bboxes[i][2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1]],
            'segmentation': mask_utils.encode(np.asfortranarray(mask))
        })
    return results


if __name__ == '__main__':
    _logger.info(f'Acne Detection and Segmentation with Mask R-CNN - {args.command}')

    # Configurations
    cfg = Config()
    _logger.info(str(cfg))

    # Devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in cfg.GPU_IDS])
    if torch.cuda.is_available() and cfg.USE_GPU:
        _device = torch.device('cuda:0')
    else:
        _device = torch.device('cpu')

    # Create model
    mrcnn = MaskRCNN(cfg)
    if cfg.USE_GPU and len(cfg.GPU_IDS) > 1:
        mrcnn = torch.nn.DataParallel(mrcnn)
    mrcnn.to(_device)

    # Select weights file to load
    if args.checkpoint:
        if args.checkpoint.lower() == "default":
            model_path = CHECKPOINT_PATH
        else:
            model_path = args.checkpoint
    else:
        model_path = ""
    # Load weights
    if model_path:
        _logger.info(f"Loading weights {model_path}")
        utils.load_weights(mrcnn, model_path)

    if args.command == 'train':
        # Training dataset
        dataset_train = AcneSegDataset(os.path.join(cfg.DATA_BASE_DIR, 'images'),
                                       os.path.join(cfg.DATA_BASE_DIR, 'annotations', 'acne_train.json'),
                                       'train', cfg, transforms(cfg.RGB_MEAN, cfg.RGB_STD, cfg.IMAGE_SHAPE[:2], 'train'))
        train_loader = DataLoader(
            dataset_train, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
        # Validation dataset
        dataset_valid = AcneSegDataset(os.path.join(cfg.DATA_BASE_DIR, 'valid_patch'),
                                       os.path.join(cfg.DATA_BASE_DIR, 'annotations', 'acne_valid.json'),
                                       'valid', cfg, transforms(cfg.RGB_MEAN, cfg.RGB_STD, cfg.IMAGE_SHAPE[:2], 'valid'))
        valid_loader = DataLoader(
            dataset_valid, cfg.BATCH_SIZE * 2, shuffle=False, num_workers=cfg.NUM_WORKERS)

        # Training - Stage 1
        _logger.info("Training network heads")
        train(mrcnn, train_loader, valid_loader,
              layers='heads',
              learning_rate=cfg.LEARNING_RATE,
              config=cfg)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        _logger.info("Fine tune Resnet stage 4 and up")
        train(mrcnn, train_loader, valid_loader,
              layers='4+',
              learning_rate=cfg.LEARNING_RATE,
              config=cfg)

        # Training - Stage 3
        # Fine tune all layers
        _logger.info("Fine tune all layers")
        train(mrcnn, train_loader, valid_loader,
              layers='all',
              learning_rate=cfg.LEARNING_RATE / 10,
              config=cfg)
    else:
        # Test dataset
        dataset_test = AcneSegDataset(os.path.join(cfg.DATA_BASE_DIR, 'test_patch'),
                                      os.path.join(cfg.DATA_BASE_DIR, 'annotations', 'acne_test.json'),
                                      'test', cfg, transforms(cfg.RGB_MEAN, cfg.RGB_STD, cfg.IMAGE_SHAPE[:2], 'test'))
        evaluate(mrcnn, dataset_test, cfg)
    _writer.close()
