import os
import os.path as osp
import numpy as np
import shutil
from sklearn.model_selection import train_test_split


def gather_obj_paths(data_dir):
    obj_paths = []
    labels = []
    classes = sorted([p for p in os.listdir(data_dir) if osp.isdir(osp.join(data_dir, p))])
    for class_name in classes:
        train_dir = osp.join(data_dir, class_name, "train")
        _obj_paths = sorted([p for p in os.listdir(train_dir) if p.endswith(".obj")])
        _labels = [class_name] * len(_obj_paths)
        obj_paths.extend(_obj_paths)
        labels.extend(_labels)
    return obj_paths, labels


def multiple_train_val_split(obj_paths, labels, num_splits, test_size):
    """
    This function uses train_test_split multiple times instead of using StratifiedShuffleSplit with n_splits=num_splits.
    This allows for additional future splits without changing the randomization of existing ones.
    """
    dataset_splits = []
    obj_paths, labels = np.asarray(obj_paths), np.asarray(labels)
    for i_split in range(num_splits):
        obj_paths_train, obj_paths_val, labels_train, labels_val = train_test_split(
            obj_paths, labels, test_size=test_size, random_state=i_split, stratify=labels)
        dataset_splits.append((obj_paths_train, obj_paths_val, labels_train, labels_val))
    return dataset_splits


def copy_subset_split_files(obj_paths, labels, data_dir, split_dir, subset_name):
    class_names = set(labels)
    for class_name in class_names:
        os.makedirs(osp.join(split_dir, class_name, subset_name))

    for obj_path, label in zip(obj_paths, labels):
        source_path = osp.join(data_dir, label, "train", obj_path)
        target_path = osp.join(split_dir, label, subset_name, obj_path)
        shutil.copyfile(source_path, target_path)


def copy_dataset_split_files(obj_paths_train, obj_paths_val, labels_train, labels_val,
                             data_dir, split_dir):
    if osp.exists(split_dir):
        shutil.rmtree(split_dir)
    copy_subset_split_files(obj_paths_train, labels_train, data_dir, split_dir, subset_name="train")
    copy_subset_split_files(obj_paths_val, labels_val, data_dir, split_dir, subset_name="test")


def split_classification_dataset(data_dir, num_splits, test_size):
    obj_paths, labels = gather_obj_paths(data_dir)
    dataset_splits = multiple_train_val_split(obj_paths, labels, num_splits, test_size)
    for i_split, dataset_split in enumerate(dataset_splits):
        data_dir_name = osp.basename(osp.normpath(data_dir))
        split_dir = osp.join(data_dir + "_val_splits", data_dir_name + "_val_split_" + str(i_split))
        obj_paths_train, obj_paths_val, labels_train, labels_val = dataset_split
        copy_dataset_split_files(obj_paths_train, obj_paths_val, labels_train, labels_val,
                                 data_dir, split_dir)


def example():
    data_dir = osp.abspath(osp.dirname(__file__) + "/../datasets/shrec_16")
    split_classification_dataset(data_dir=data_dir,
                                 num_splits=4,
                                 test_size=0.25)


if __name__ == '__main__':
    example()
