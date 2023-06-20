import os
from PIL import Image

input_img_root = '/home/lihaoyang/dataset/cls/RCTW-17'

all_img_names = os.listdir(input_img_root)
save_num_img = 3
for img_name in all_img_names[:save_num_img]:
    # img_name = all_img_names[0]
    img_path = os.path.join(input_img_root, img_name)
    im = Image.open(img_path)   

    angle = 30
    rot_im = im.rotate(angle, expand=False)  # counterclockwise
    rot_im_expand = im.rotate(angle, expand=True)  # counterclockwise

    print(f'\n--------{img_name} shape-------')
    print('im:', im.size)
    print('rot_im:', rot_im.size)
    print('rot_im_expand:', rot_im_expand.size)

    im.save(os.path.join('/home/lihaoyang/dataset/cls/rotate_test', img_name))
    rot_im.save(os.path.join('/home/lihaoyang/dataset/cls/rotate_test', f'{angle}_{img_name}'))
    rot_im_expand.save(os.path.join('/home/lihaoyang/dataset/cls/rotate_test', f'expand_{angle}_{img_name}'))