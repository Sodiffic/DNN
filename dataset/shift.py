import glob
import os
import shutil
"""把val里的一些测试用例挪到test文件夹里"""
file_path = 'F:/ntu_120/val'
save_path = 'F:/ntu_120/test/'
sample_file = glob.glob(f'{file_path}/*.npy', recursive=True)
index = -1
num = 0
for path in sample_file:
    _, filename = os.path.split(path)
    cat = filename.split('_')[0]  # 类别名
    if cat == index and num>=5:
        continue
    elif cat !=index:
        index=cat
        shutil.copy(path, save_path+filename)
        print(path)
        num=1
    else:
        shutil.copy(path, save_path+filename)
        print(path)
        num += 1






