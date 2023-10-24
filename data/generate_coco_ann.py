import os
import datetime
import json
import numpy as np

base_dir = './ACNE'
ann_dir = os.path.join(base_dir, 'annotations')
datasets = ['train', 'valid', 'test']

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
    # bbox shape: [x1, y1, width, height]
    return np.array([top_left[0], top_left[1],
                     bottom_right[0] + 1 - top_left[0], bottom_right[1] + 1 - top_left[1]])


for dataset in datasets:
    print(f'creating {dataset}')
    with open(os.path.join(base_dir, f'{dataset}_list.txt'), 'r', encoding='utf8') as fp:
        filelist = fp.read().split('\n')
    ann_id = 0
    for idx, file in enumerate(filelist):
        print(file)
        with open(os.path.join(ann_dir, file + '.json'), 'r', encoding='utf8') as fp:
            ann = json.load(fp)
        image = {
            'id': idx,
            'width': ann['imageWidth'],
            'height': ann['imageHeight'],
            'file_name': file + '.jpg',
        }
        base_obj['images'].append(image)
        for shape in ann['shapes']:
            points = np.array(shape['points'], dtype=np.float32).round()
            box = points_to_bbox(points).tolist()
            base_obj['annotations'].append({
                'id': ann_id,
                'image_id': idx,
                'category_id': category_to_id[shape['label']],
                'segmentation': [points.flatten().tolist()],
                'area': box[2] * box[3],
                'bbox': box,
                'iscrowd': 0
            })
            ann_id += 1

    with open(os.path.join(ann_dir, f'acne_{dataset}.json'), 'w', encoding='utf8') as fp:
        json.dump(base_obj, fp)

    base_obj['images'] = []
    base_obj['annotations'] = []
