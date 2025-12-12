import torchvision
import os
from PIL import Image
import numpy as np
import pickle
from tqdm import trange
from os.path import join
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

parser = argparse.ArgumentParser("generate_cifar10_dataset")
parser.add_argument('--save-dir', type=str, help='The directory to save the cifar-10 dataset')
parser.add_argument('--dataset_save_dir', type=str, help='The directory to save the image dataset')
args = parser.parse_args()

if __name__ == '__main__':
    save_dir = args.save_dir
    dst_dir = args.dataset_save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 下载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(root=save_dir, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=save_dir, train=False, download=True)

    print(f"CIFAR-10 is downloaded to: {save_dir}")

    # 获取 CIFAR-10 类别名称
    class_names = trainset.classes  # ['airplane', 'automobile', 'bird', ..., 'truck']
    label_to_class_mapping = {name: format_int_to_str(i) for i, name in enumerate(class_names)}

    # 创建目标数据集目录
    my_mkdirs(dst_dir)

    for data_set, dataset in [('train', trainset), ('test', testset)]:
        print(f"Processing {data_set} dataset...")
        my_mkdirs(join(dst_dir, data_set))

        # 按类别创建文件夹
        for class_name in class_names:
            class_id = label_to_class_mapping[class_name]
            my_mkdirs(join(dst_dir, data_set, class_id))

        # 遍历数据集并保存图片
        for i in trange(len(dataset)):
            img, label = dataset[i]
            class_id = label_to_class_mapping[class_names[label]]
            img.save(join(dst_dir, data_set, class_id, f"{i:06}.png"))

    print("All done.")
