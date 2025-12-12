import torchvision
import os
from PIL import Image
import numpy as np
import pickle
import os
from tqdm import trange
from os.path import join
import torchvision
import argparse

def format_int_to_str(number):
    return "{:05}".format(number)

def my_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

parser = argparse.ArgumentParser("generate_cifa_dataset")
parser.add_argument('--save-dir', type=str, default='/l/users/jiacheng.cui/Nas_DD_data/dataset',
                    help='The directory to save the CIFAR-100 dataset')
parser.add_argument('--dataset_save_dir', type=str, default='/l/users/jiacheng.cui/Nas_DD_data/dataset/cifar100',
                    help='The directory to save the image dataset')
args = parser.parse_args()

if __name__ == '__main__':
    save_dir = args.save_dir
    src_dir = join(save_dir, 'cifar-100-python')
    dst_dir = args.dataset_save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    trainset = torchvision.datasets.CIFAR100(
        root=save_dir,
        train=True,
        download=True
    )

    testset = torchvision.datasets.CIFAR100(
        root=save_dir,
        train=False,
        download=True
    )

    print(f"CIFAR-100 is downloaded to: {save_dir}")
    
    meta = unpickle(join(src_dir, 'meta')) # KEYS: {'fine_label_names', 'coarse_label_names'}
    my_mkdirs(dst_dir)

    for data_set in ['train', 'test']:
        print('Unpickling {} dataset......'.format(data_set))
        data_dict = unpickle(join(src_dir, data_set)) # KEYS: {'filenames', 'batch_label', 'fine_labels', 'coarse_labels', 'data'}
        my_mkdirs(join(dst_dir, data_set))

        for fine_label_name in meta['fine_label_names']:
            my_mkdirs(join(dst_dir, data_set, fine_label_name))

        for i in trange(data_dict['data'].shape[0]):
            img = np.reshape(data_dict['data'][i], (3, 32, 32))
            i0 = Image.fromarray(img[0])
            i1 = Image.fromarray(img[1])
            i2 = Image.fromarray(img[2])
            img = Image.merge('RGB', (i0, i1, i2))
            img.save(join(dst_dir, data_set, meta['fine_label_names'][data_dict['fine_labels'][i]], data_dict['filenames'][i]))

    print('All done.')
    
import os
import torchvision

# CIFAR-100 标签与类别名映射
trainset = torchvision.datasets.CIFAR100(root=save_dir, train=True, download=True)
label_to_class_mapping = {class_name: i for i, class_name in enumerate(trainset.classes)}

def rename_subfolders(root_dir):
    # 循环遍历根目录下的所有子文件夹
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # 查找子文件夹名对应的 CIFAR-100 标签
            if subfolder in label_to_class_mapping:
                label = label_to_class_mapping[subfolder]
                # 构造新的文件夹名称
                new_folder_name = str(format_int_to_str(label))
                new_folder_path = os.path.join(root_dir, new_folder_name)
                # 重命名文件夹
                os.rename(subfolder_path, new_folder_path)
            else:
                print(f"Warning: Subfolder '{subfolder}' not found in CIFAR-100 class mapping")


# 使用该函数遍历并重命名指定路径下的子文件夹中的文件
root_directory = os.path.join(dst_dir,"train")  # 替换为实际的路径
rename_subfolders(root_directory)

root_directory = os.path.join(dst_dir,"test")  # 替换为实际的路径
rename_subfolders(root_directory)