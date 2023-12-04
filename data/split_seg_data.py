import os
import random
import shutil

random.seed(1024)

base_dir = './ACNE/'
dest_dir = './ACNE_seg/'
dest_img_dir = './ACNE_seg/images/'
ann_dir = os.path.join(base_dir, 'annotations')
filenames = [filename[:-5]
             for filename in os.listdir(ann_dir)
             if filename[-4:] == 'json']

print(f'Total data: {len(filenames)}')

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir + '/images/')
for filename in filenames:
    if os.path.exists(base_dir+'images/'+filename+'.jpg'):
        shutil.copy(base_dir+'images/'+filename+'.jpg',
                    dest_img_dir+filename+'.jpg')
    else:
        print(filename)

test_list = random.sample(filenames, 60)
train_list = [filename for filename in filenames if filename not in test_list]
valid_list = random.sample(train_list, 16)
train_list = [filename for filename in train_list if filename not in valid_list]
print(f'Train Samples: {len(train_list)}')
print(f'Valid Samples: {len(valid_list)}')
print(f'Test  Samples: {len(test_list)}')

with open(os.path.join(dest_dir, 'train_list.txt'), 'w', encoding='utf8') as fp:
    fp.write('\n'.join(train_list))

with open(os.path.join(dest_dir, 'valid_list.txt'), 'w', encoding='utf8') as fp:
    fp.write('\n'.join(valid_list))

with open(os.path.join(dest_dir, 'test_list.txt'), 'w', encoding='utf8') as fp:
    fp.write('\n'.join(test_list))
