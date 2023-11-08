import os
import copy
import datetime
import json
import numpy as np
import skimage.io as io

base_dir = './ACNE'
img_dir = os.path.join(base_dir, 'images')
ann_dir = os.path.join(base_dir, 'annotations')

today = datetime.date.today()
base_obj = {
    'info': {
        'year': today.year,
        'version': '1.0.0',
        'description': '',
        'contributor': '',
        'url': '',
        'date_created': '',
    },
    'images': [],
    'annotations': [],
    'categories': [],
    'license': [],
}
categories = ['papule', 'nevus', 'nodule',
              'open_comedo', 'closed_comedo',
              'atrophic_scar', 'hypertrophic_scar',
              'melasma', 'pustule', 'other']
category_to_id = {c: i for i, c in enumerate(categories, 1)}
for i, cls in enumerate(categories, 1):
    base_obj['categories'].append({
        'id': i,
        'name': cls,
        'supercategory': '',
    })


def points_to_bbox(points):
    top_left = np.min(points, axis=0)
    bottom_right = np.max(points, axis=0)
    # bbox shape: [x1, y1, x2, y2]
    return np.array([top_left[0] - 0.5, top_left[1] - 0.5,
                     bottom_right[0] + 0.5, bottom_right[1] + 0.5])


def clip_box(box, window):
    box[[0, 2]] = box[[0, 2]].clip(window[0], window[2])
    box[[1, 3]] = box[[1, 3]].clip(window[1], window[3])
    return box


def clip_shape(points, window):
    points[:, 0] = points[:, 0].clip(window[0], window[2])
    points[:, 1] = points[:, 1].clip(window[1], window[3])
    return points


ann_id = 0
img_id = 0
MIN_AREA = 50
MIN_WIDTH = 5
MIN_HEIGHT = 5

print(f'creating train data annotation')
train_obj = copy.deepcopy(base_obj)
with open(os.path.join(base_dir, 'train_list.txt'), 'r', encoding='utf8') as fp:
    filelist = fp.read().split('\n')
for file in filelist:
    print(file)
    with open(os.path.join(ann_dir, file + '.json'), 'r', encoding='utf8') as fp:
        ann = json.load(fp)
    image = {
        'id': img_id,
        'width': ann['imageWidth'],
        'height': ann['imageHeight'],
        'file_name': file + '.jpg',
    }
    train_obj['images'].append(image)
    window = [0, 0, ann['imageWidth'], ann['imageHeight']]
    for shape in ann['shapes']:
        points = np.array(shape['points'])
        box = points_to_bbox(points)
        points = clip_shape(points, window)
        box = clip_box(box, window)
        box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        area = box[2] * box[3]
        if area > MIN_AREA and box[2] > MIN_WIDTH and box[3] > MIN_HEIGHT:
            train_obj['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': category_to_id[shape['label']],
                'segmentation': [points.flatten().tolist()],
                'area': area,
                'bbox': box,
                'iscrowd': 0
            })
            ann_id += 1
    img_id += 1
with open(os.path.join(ann_dir, 'acne_train.json'), 'w', encoding='utf8') as fp:
    json.dump(train_obj, fp)


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


WIN_SIZE = [1024, 1024]
WIN_STRIDE = [960, 960]

print(f'creating valid data annotation')
valid_save_dir = os.path.join(base_dir, 'valid_patch')
if not os.path.exists(valid_save_dir):
    os.makedirs(valid_save_dir)
valid_obj = copy.deepcopy(base_obj)
with open(os.path.join(base_dir, 'valid_list.txt'), 'r', encoding='utf8') as fp:
    filelist = fp.read().split('\n')
for file in filelist:
    print(file)
    with open(os.path.join(ann_dir, file + '.json'), 'r', encoding='utf8') as fp:
        ann = json.load(fp)
    image = io.imread(os.path.join(img_dir, file + '.jpg'))
    win_gen = WindowGenerator(ann['imageHeight'], ann['imageWidth'],
                              WIN_SIZE[0], WIN_SIZE[1], WIN_STRIDE[0], WIN_STRIDE[1])

    for h_slice, w_slice in win_gen:
        patch = image[h_slice, w_slice]
        io.imsave(os.path.join(valid_save_dir, f'{img_id:08d}.jpg'), patch)
        valid_obj['images'].append({
            'id': img_id,
            'width': 1024,
            'height': 1024,
            'file_name': f'{img_id:08d}.jpg',
            'source_file': file + '.jpg',
            'meta': [ann['imageHeight'], ann['imageWidth'],
                     w_slice.start, h_slice.start, w_slice.stop, h_slice.stop]
        })

        window = [w_slice.start, h_slice.start, w_slice.stop, h_slice.stop]
        for shape in ann['shapes']:
            points = np.array(shape['points'])
            box = points_to_bbox(points)
            points = clip_shape(points, window)
            box = clip_box(box, window)
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            area = box[2] * box[3]
            if area > MIN_AREA and box[2] > MIN_WIDTH and box[3] > MIN_HEIGHT:
                points[:, 0] -= window[0]
                points[:, 1] -= window[1]
                box[0] -= window[0]
                box[1] -= window[1]
                valid_obj['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': category_to_id[shape['label']],
                    'segmentation': [points.flatten().tolist()],
                    'area': area,
                    'bbox': box,
                    'iscrowd': 0
                })
                ann_id += 1
        img_id += 1
with open(os.path.join(ann_dir, 'acne_valid.json'), 'w', encoding='utf8') as fp:
    json.dump(valid_obj, fp)

print(f'creating test data annotation')
test_save_dir = os.path.join(base_dir, 'test_patch')
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)
test_obj = copy.deepcopy(base_obj)
with open(os.path.join(base_dir, 'test_list.txt'), 'r', encoding='utf8') as fp:
    filelist = fp.read().split('\n')
for file in filelist:
    print(file)
    with open(os.path.join(ann_dir, file + '.json'), 'r', encoding='utf8') as fp:
        ann = json.load(fp)
    image = io.imread(os.path.join(img_dir, file + '.jpg'))
    win_gen = WindowGenerator(ann['imageHeight'], ann['imageWidth'],
                              WIN_SIZE[0], WIN_SIZE[1], WIN_STRIDE[0], WIN_STRIDE[1])

    for h_slice, w_slice in win_gen:
        patch = image[h_slice, w_slice]
        io.imsave(os.path.join(test_save_dir, f'{img_id:08d}.jpg'), patch)
        test_obj['images'].append({
            'id': img_id,
            'width': 1024,
            'height': 1024,
            'file_name': f'{img_id:08d}.jpg',
            'source_file': file + '.jpg',
            'meta': [ann['imageHeight'], ann['imageWidth'],
                     w_slice.start, h_slice.start, w_slice.stop, h_slice.stop]
        })

        window = [w_slice.start, h_slice.start, w_slice.stop, h_slice.stop]
        for shape in ann['shapes']:
            points = np.array(shape['points'])
            box = points_to_bbox(points)
            points = clip_shape(points, window)
            box = clip_box(box, window)
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            area = box[2] * box[3]
            if area > MIN_AREA and box[2] > MIN_WIDTH and box[3] > MIN_HEIGHT:
                points[:, 0] -= window[0]
                points[:, 1] -= window[1]
                box[0] -= window[0]
                box[1] -= window[1]
                test_obj['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': category_to_id[shape['label']],
                    'segmentation': [points.flatten().tolist()],
                    'area': area,
                    'bbox': box,
                    'iscrowd': 0
                })
                ann_id += 1
        img_id += 1
with open(os.path.join(ann_dir, 'acne_test.json'), 'w', encoding='utf8') as fp:
    json.dump(test_obj, fp)

print('Done!')
