import open3d as o3d
import numpy as np


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
