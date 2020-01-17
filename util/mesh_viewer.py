import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import matplotlib.cm as cm
import pylab as pl
import numpy as np
import os

V = np.array
r2h = lambda x: colors.rgb2hex(tuple(map(lambda y: y / 255., x)))
surface_color = r2h((225, 225, 225))
edge_color = r2h((90, 90, 90))




def init_plot():
    ax = pl.figure().add_subplot(111, projection='3d')
    # hide axis, thank to
    # https://stackoverflow.com/questions/29041326/3d-plot-with-matplotlib-hide-axes-but-keep-axis-labels/
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return (ax, [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf])


def update_lim(mesh, plot):
    vs = mesh[0]
    for i in range(3):
        plot[1][2 * i] = min(plot[1][2 * i], vs[:, i].min())
        plot[1][2 * i + 1] = max(plot[1][2 * i], vs[:, i].max())
    return plot


def update_plot(mesh, plot):
    if plot is None:
        plot = init_plot()
    return update_lim(mesh, plot)


def surfaces(mesh, plot):
    vs, faces, edges = mesh
    vtx = vs[faces]
    edgecolor = edge_color if not len(edges) else 'none'
    tri = a3.art3d.Poly3DCollection(vtx, facecolors=surface_color +'55', edgecolors=edgecolor,
                                    linewidths=.8)#, linestyles='dashdot')
    plot[0].add_collection3d(tri)
    return plot


edges_values_map = "sqrt"  # sort/log2/sqrt/else - original values


def segments(mesh, plot):
    vs, _, edges = mesh
    if edges_values_map == "sort":
        for i, (_, edge_idx) in enumerate(sorted(edges, key=lambda x: x[0])):
            edge = vs[edge_idx]
            line = a3.art3d.Line3DCollection([edge], linewidths=.8)
            line.set_color(cm.viridis((i+1) / len(edges)))
            plot[0].add_collection3d(line)
    else:
        for (edge_c, edge_idx) in edges:
            edge = vs[edge_idx]
            line = a3.art3d.Line3DCollection([edge], linewidths=.8)
            if edges_values_map == "sqrt":
                line.set_color(cm.viridis(np.sqrt(edge_c)))
            elif edges_values_map == "log2":
                line.set_color(cm.viridis(np.log2(1 + edge_c)))
            else:
                line.set_color(cm.viridis(edge_c))
            plot[0].add_collection3d(line)
    return plot


def plot_mesh(mesh, *whats, plot=None, out_path=None):
    for what in [update_plot] + list(whats):
        plot = what(mesh, plot)
    if out_path is not None:
        pl.savefig(out_path)


def parse_obje(obj_file, scale_by):
    vs = []
    faces = []
    edges = []

    def add_to_edges():
        edges.append((edge_c, edge_v))

    def fix_vertices():
        nonlocal vs, scale_by
        vs = V(vs)
        z = vs[:, 2].copy()
        vs[:, 2] = vs[:, 1]
        vs[:, 1] = z
        max_range = 0
        for i in range(3):
            min_value = np.min(vs[:, i])
            max_value = np.max(vs[:, i])
            max_range = max(max_range, max_value - min_value)
            vs[:, i] -= min_value
        if not scale_by:
            scale_by = max_range
        vs /= scale_by

    with open(obj_file) as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:]])
            elif splitted_line[0] == 'f':
                faces.append([int(c) - 1 for c in splitted_line[1:]])
            elif splitted_line[0] == 'e':
                if len(splitted_line) >= 4:
                    edge_v = [int(c) - 1 for c in splitted_line[1:-1]]
                    edge_c = float(splitted_line[-1])
                    add_to_edges()

    vs = V(vs)
    fix_vertices()
    faces = V(faces, dtype=int)
    return (vs, faces, edges), scale_by


def view_meshes(args, files):
    scale = 0
    for file in files:
        in_file_path = os.path.join(args.indir, file)
        out_file_path = os.path.splitext(os.path.join(args.outdir, file))[0] + ".jpg"
        mesh, scale = parse_obje(in_file_path, scale)
        plot_mesh(mesh, surfaces, segments, plot=None, out_path=out_file_path)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser("view meshes")
    parser.add_argument('--indir', default=None, type=str)
    parser.add_argument('--outdir', default=None, type=str)
    args = parser.parse_args()

    if args.indir is None or args.outdir is None:
        print("wrong command line params")
    else:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        files = [file for file in os.listdir(args.indir) if file.endswith(".obj")]
        # view meshes
        view_meshes(args, files)

