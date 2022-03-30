import open3d as o3d
import numpy as np
from file_io import read_pv_file

def visualize_binary(pcl, mask):
    points = np.asarray(pcl.points)
    color_points = [[0.0, 0.0, 1.0] for i in range(points.shape[0])]
    for i, bool in enumerate(mask):
        if not bool:
            color_points[i] = [1.0, 0.0, 0.0]
    pcl.colors = o3d.utility.Vector3dVector(color_points)
    o3d.visualization.draw_geometries([pcl])


def line_set(start, stop):
        num = start.shape[0]
        assert num == stop.shape[0], "number of elements of start array must be equal to number of elements of stop"
        lines = [[i, num+i] for i in range(num)]
        points = np.concatenate([start, stop], axis=0)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        return line_set

def visualize_hypo(input_file):
    points, vecs, scalar, nocam = read_pv_file(input_file)
    end = points + vecs
    ls = line_set(points, end)
    return ls

# if __name__=="__main__":
#     o3d.visualization.draw_geometries([visualize_hypo("/home/flexsight/Downloads/dtu/GT__/scan116_train.npz")])