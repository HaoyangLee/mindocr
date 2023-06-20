import os
import shutil

"""mindcv data format
root = '/home/lihaoyang/dataset/cls'
src_roots = ['RCTW17_mindcv', 'MTWI_mindcv', 'LSVT_mindcv']
dst_root = os.path.join(root, 'merged_data_mindcv')

if os.path.exists(dst_root):
    shutil.rmtree(dst_root)

for s in src_roots:
    src_root = os.path.join(root, s)
    dst_root = os.path.join(root, 'merged_data_mindcv')
    shutil.copytree(src_root, dst_root)
"""


# """mindocr data format
def add_imgdir_to_anno(data_name, src, dst):
    fr = open(src, 'r')
    fw = open(dst, 'a')

    for line_read in fr.readlines():
        fw.write(f'{data_name}_ori_and_rot/' + line_read)

    fr.close()
    fw.close()

root = '/home/lihaoyang/dataset/cls'
data_names = ['RCTW17', 'MTWI', 'LSVT']
stages = ['train', 'val', 'eval']

for s in stages:
    dst = os.path.join(root, 'anno_merged_data', f'anno_merged_data_{s}.txt')
    for dn in data_names:
        src = os.path.join(root, f'anno_{dn}', f'anno_{dn}_{s}.txt')
        add_imgdir_to_anno(dn, src, dst)

# """