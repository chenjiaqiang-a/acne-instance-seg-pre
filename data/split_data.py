import os
import random

random.seed(1024)

base_dir = './ACNE'
ann_dir = os.path.join(base_dir, 'annotations')
filenames = [filename[:-5]
             for filename in os.listdir(ann_dir)
             if filename[-4:] == 'json']

print(f'Total data: {len(filenames)}')

test_list = random.sample(filenames, int(len(filenames)/5))
train_list = [filename for filename in filenames if filename not in test_list]
valid_list = random.sample(train_list, int(len(train_list)/10))
train_list = [filename for filename in train_list if filename not in valid_list]
print(f'Train Samples: {len(train_list)}')
print(f'Valid Samples: {len(valid_list)}')
print(f'Test  Samples: {len(test_list)}')

with open(os.path.join(base_dir, 'train_list.txt'), 'w', encoding='utf8') as fp:
    fp.write('\n'.join(train_list))

with open(os.path.join(base_dir, 'valid_list.txt'), 'w', encoding='utf8') as fp:
    fp.write('\n'.join(valid_list))

with open(os.path.join(base_dir, 'test_list.txt'), 'w', encoding='utf8') as fp:
    fp.write('\n'.join(test_list))
