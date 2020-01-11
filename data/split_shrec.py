import os
import os.path as osp
import shutil
from numpy.random import RandomState
import hashlib


def split_shrec(data_dir, out_dir, samples_per_split, n_splits):
    """
    out_dir must be an absolute path. Normally, it would be data_dir's parent
    """
    classes = sorted([p for p in os.listdir(data_dir) if osp.isdir(osp.join(data_dir, p))])
    for cls in classes:
        test_objs = [p for p in os.listdir(osp.join(data_dir, cls, "test")) if p.endswith(".obj")]
        train_objs = sorted([p for p in os.listdir(osp.join(data_dir, cls, "train")) if p.endswith(".obj")])
        total_samples = len(train_objs)
        for i_split in range(n_splits):
            split_dir = osp.join(out_dir, "shrec_{}_split_{}".format(samples_per_split, i_split))
            dirs_to_make = [osp.join(split_dir, cls, "train"), osp.join(split_dir, cls, "test")]
            for _dir in dirs_to_make:
                if osp.exists(_dir):
                    shutil.rmtree(_dir)
                os.makedirs(_dir)

            unique_id = ",".join([cls, str(i_split), str(n_splits), str(samples_per_split)])
            random_seed = int(hashlib.md5(unique_id.encode()).hexdigest()[:8], base=16)
            sample_inds = RandomState(random_seed).permutation(total_samples)[:samples_per_split]
            print(unique_id, [train_objs[i] for i in sample_inds])

            for i_sample in sample_inds:
                src = osp.join(data_dir, cls, "train", train_objs[i_sample])
                dst = osp.join(split_dir, cls, "train", train_objs[i_sample])
                shutil.copyfile(src, dst)

            for test_obj in test_objs:
                src = osp.join(data_dir, cls, "test", test_obj)
                dst = osp.join(split_dir, cls, "test", test_obj)
                shutil.copyfile(src, dst)


if __name__ == '__main__':
    split_shrec(data_dir="/home/tomer/deploys/MeshCNN/datasets/shrec_16",
                out_dir="/home/tomer/deploys/MeshCNN/datasets",
                samples_per_split=10,
                n_splits=3)
    # split_shrec(data_dir=r"C:\Users\user\dev\MeshCNN\datasets\shrec_16",
    #             out_dir=r"C:\Users\user\dev\MeshCNN\datasets",
    #             samples_per_split=10,
    #             n_splits=3)
