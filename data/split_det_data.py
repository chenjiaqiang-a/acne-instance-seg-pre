import os
import random

random.seed(1024)

base_dir = './ACNE_det/'
src_ann_dir = './ACNE/annotations/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

filenames = [filename[:-5]
             for filename in os.listdir(src_ann_dir)
             if filename[-4:] == 'json']

print(f'Total data: {len(filenames)}')

duplicate = ['陈荣荣', '代自豪', '张艺潇', '代敏', '刘然', '张耀',
             '何宇航', '任鹏锦', '张继泽', '安妮', '屈艳', '康禾', 
             '刘栩男', '岳钰婷', '谢柠枍', '张藐', '徐婧', '李陈']

file_list = [filename for filename in filenames if filename.split('_')[2] not in duplicate]
test_list = random.sample(file_list, 50)

file_list = [filename for filename in file_list if filename not in test_list]
valid_list = random.sample(file_list, 26)

train_list = [filename for filename in filenames
              if filename not in test_list and filename not in valid_list]

print(f'Train Samples: {len(train_list)}')
print(f'Valid Samples: {len(valid_list)}')
print(f'Test  Samples: {len(test_list)}')

with open(os.path.join(base_dir, 'train_list.txt'), 'w', encoding='utf8') as fp:
    fp.write('\n'.join(train_list))

with open(os.path.join(base_dir, 'valid_list.txt'), 'w', encoding='utf8') as fp:
    fp.write('\n'.join(valid_list))

with open(os.path.join(base_dir, 'test_list.txt'), 'w', encoding='utf8') as fp:
    fp.write('\n'.join(test_list))
