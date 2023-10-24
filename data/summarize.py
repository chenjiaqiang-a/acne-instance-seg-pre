import os
import json

base_dir = './ACNE'
ann_dir = os.path.join(base_dir, 'annotations')
filenames = [filename
             for filename in os.listdir(ann_dir)
             if filename[-4:] == 'json']
print(f'Total samples: {len(filenames)}')

categories = []
counter = {}

for filename in filenames:
    with open(os.path.join(ann_dir, filename), 'r', encoding='utf8') as fp:
        obj = json.load(fp)
        for shape in obj['shapes']:
            if shape['label'] in categories:
                counter[shape['label']] += 1
            else:
                categories.append(shape['label'])
                counter[shape['label']] = 1

print('Categories: number')
for key in counter:
    print(f'{key}: {counter[key]}')
