import os
import re
import torch
import numpy as np


############################################################
#  Model
############################################################

def load_weights(model, filepath):
    if os.path.exists(filepath):
        state_dict = torch.load(filepath)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Weight file not found ...")


def set_trainable(model, layer_regex):
    """Sets model layers as trainable if their names match
    the given regular expression.
    """
    for param in model.named_parameters():
        layer_name = param[0]
        trainable = bool(re.fullmatch(layer_regex, layer_name))
        if not trainable:
            param[1].requires_grad = False


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (x, y) and a list of (w, h)
    box_centers = np.stack(
        [box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (x1, y1, x2, y2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (x1, y1, x2, y2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors([scales[i]], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Bounding Boxes
############################################################

def box_refinement(references, targets):
    """Compute refinement needed to transform references to targets.
    references and targets are [N, (x1, y1, x2, y2)]
    """
    shape = references.size()
    references = references.reshape((-1, 4))
    targets = targets.reshape((-1, 4))

    width = references[:, 2] - references[:, 0]
    height = references[:, 3] - references[:, 1]
    center_x = references[:, 0] + 0.5 * width
    center_y = references[:, 1] + 0.5 * height

    t_width = targets[:, 2] - targets[:, 0]
    t_height = targets[:, 3] - targets[:, 1]
    t_center_x = targets[:, 0] + 0.5 * t_width
    t_center_y = targets[:, 1] + 0.5 * t_height

    dx = (t_center_x - center_x) / width
    dy = (t_center_y - center_y) / height
    dw = torch.log(t_width / width)
    dh = torch.log(t_height / height)

    deltas = torch.stack([dx, dy, dw, dh], dim=1)
    deltas = deltas.reshape(shape)
    return deltas


def apply_box_deltas(references, deltas):
    """Applies the given deltas to the given boxes.
    references: [N, 4] where each row is x1, y1, x2, y2
    deltas: [N, 4] where each row is [dx, dy, log(dw), log(dh)]
    """
    shape = references.size()
    references = references.reshape((-1, 4))
    deltas = deltas.reshape((-1, 4))
    # Convert to x, y, w, h
    width = references[:, 2] - references[:, 0]
    height = references[:, 3] - references[:, 1]
    center_x = references[:, 0] + 0.5 * width
    center_y = references[:, 1] + 0.5 * height
    # Apply deltas
    t_center_x = center_x + deltas[:, 0] * width
    t_center_y = center_y + deltas[:, 1] * height
    t_width = width * torch.exp(deltas[:, 2])
    t_height = height * torch.exp(deltas[:, 3])
    # Convert back to x1, y1, x2, y2
    x1 = t_center_x - 0.5 * t_width
    y1 = t_center_y - 0.5 * t_height
    x2 = x1 + t_width
    y2 = y1 + t_height
    targets = torch.stack([x1, y1, x2, y2], dim=1)

    targets = targets.reshape(shape)
    return targets


def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is x1, y1, x2, y2
    window: [4] in the form x1, y1, x2, y2
    """
    shape = boxes.size()
    boxes = boxes.reshape((-1, 4))
    x1 = boxes[:, 0:1].clamp(float(window[0]), float(window[2]))
    y1 = boxes[:, 1:2].clamp(float(window[1]), float(window[3]))
    x2 = boxes[:, 2:3].clamp(float(window[0]), float(window[2]))
    y2 = boxes[:, 3:4].clamp(float(window[1]), float(window[3]))
    boxes = torch.cat([x1, y1, x2, y2], dim=1)
    boxes = boxes.reshape(shape)
    return boxes


############################################################
#  Inference
############################################################

class WindowGenerator:
    def __init__(self, h, w, ch, cw, si=1, sj=1):
        if h < ch or w < cw:
            raise ValueError(f'`h` must greater than `ch` and `w` must greater than `cw`,'
                             f'but got `h` = {h} `ch` = {ch} and `w` = {w} `cw` = {cw}')
        self.h, self.w = h, w
        self.ch, self.cw = ch, cw
        self.si, self.sj = si, sj
        self._i, self._j = 0, 0

    def __next__(self):
        if self._i > self.h:
            raise StopIteration

        bottom = min(self._i + self.ch, self.h)
        right = min(self._j + self.cw, self.w)
        top = max(0, bottom - self.ch)
        left = max(0, right - self.cw)

        if self._j >= self.w - self.cw:
            if self._i >= self.h - self.ch:
                self._i = self.h + 1
            self._next_row()
        else:
            self._j += self.sj
            if self._j > self.w:
                self._next_row()

        return slice(top, bottom, 1), slice(left, right, 1)

    def _next_row(self):
        self._i += self.si
        self._j = 0

    def __iter__(self):
        return self
