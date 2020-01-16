import os
import os.path as osp
import pandas as pd

data_dir = r"C:\Users\user\dev\MeshCNN\datasets\human_seg"
df_lines = []
for obj_dir in [osp.join(data_dir, "train"), osp.join(data_dir, "test")]:
    for p in [p for p in os.listdir(obj_dir) if p.endswith(".obj")]:
        obj_path = osp.join(obj_dir, p)
        with open(obj_path, 'r') as f:
            txt = f.read()
        edges = []
        for line in txt.split('\n'):
            if line.startswith('f'):
                v = line[2:].split(' ')
                edges.append((v[0], v[1]))
                edges.append((v[1], v[2]))
                edges.append((v[0], v[2]))
        n_edges = len(set(edges))
        df_lines.append([obj_path, n_edges])
df = pd.DataFrame(data=df_lines, columns=["path", "n_edges"])