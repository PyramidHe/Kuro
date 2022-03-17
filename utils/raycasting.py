import open3d as o3d
import open3d.core as o3c
import numpy as np


def are_visible(intrinsics, extrinsics, resolution, mesh, points, eps=0.001):
    """
    Given input camera parameters, resolution, mesh and points sampled on the surface of the mesh get an output vector
    which tell for each point if it is visible by the camera
         Parameters:
            intrinsics (numpy.ndarray[numpy.float64[3, 3]]): intrinsic matrix
            extrinsics (numpy.ndarray[numpy.float64[4, 4]]): extrinsic matrix
            resolution (tuple(int, int)): resolution of camera
            mesh (open3d.geometry.PointCloud): input mesh
            points (numpy.ndarray[numpy.float64[n, 3]]): point sampled on the surface of the mesh
            eps (float): error threshold, values too low will filter out good points, while higher values could not
                filter bad points
        Returns:
            visible (numpy.ndarray[numpy.bool[n]]): vector of boolean values which describe which point is visible
    """
    cam_mat = extrinsics.copy()

    rot = cam_mat[:3, :3]
    inv_rot = np.linalg.inv(rot)
    cam_point = -np.matmul(inv_rot, cam_mat[:3, 3])

    cam_mat[:3, :4] = np.matmul(intrinsics, cam_mat[:3, :4])

    hom_ones = np.ones((points.shape[0], 1))
    hom_points = np.concatenate((points, hom_ones), axis=1)
    points2d = np.transpose(np.matmul(cam_mat, np.transpose(hom_points)))
    points2d = points2d/np.expand_dims(points2d[:, 2], axis=-1)
    mask_x = (points2d[:, 0] < resolution[0])*(points2d[:, 0] > 0.0)
    mask_y = (points2d[:, 1] < resolution[1]) * (points2d[:, 1] > 0.0)
    visible = mask_x * mask_y

    if not (mesh==None):
        cam_point = np.transpose(np.repeat(np.expand_dims(cam_point, axis=-1), points.shape[0], axis=1))
        vectors = points - cam_point
        vec_norm = np.expand_dims(np.linalg.norm(vectors, axis=1), axis=-1)
        vectors = vectors/vec_norm
        rays = np.concatenate((cam_point, vectors), axis=1).astype(np.float32)
        dists = np.expand_dims(raycast(mesh, rays)['t_hit'].numpy(), axis=-1)
        reconstructed_points = cam_point + vectors * dists
        err = np.linalg.norm(points-reconstructed_points, axis=1)
        mask_hit = err < eps
        visible = mask_hit * visible

    return visible


def sample_visible(intrinsics, extrinsics, resolution, mesh, number_of_points,  eps=0.001):
    """
    Given input camera parameters, resolution, mesh and points sampled on the surface of the mesh get an output vector
    which tell for each point if it is visible by the camera
         Parameters:
            intrinsics (numpy.ndarray[numpy.float64[3, 3]]): intrinsic matrix
            extrinsics (numpy.ndarray[numpy.float64[4, 4]]): extrinsic matrix
            resolution (tuple(int, int)): resolution of camera
            mesh (open3d.geometry.PointCloud): input mesh
            number_of_points (int): number of point to sample on the surface mesh
            eps (float): error threshold, value too low will filter out good points, while higher values could not
                filter bad points
        Returns:
             points (numpy.ndarray[numpy.float64[n, 3]]): point sampled on the surface of the mesh
             visible (numpy.ndarray[numpy.bool[n]]): vector of boolean values which describe which point is visible
    """
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    points = np.asarray(pcd.points)
    visible = are_visible(intrinsics, extrinsics, resolution, mesh, points, eps=0.001)
    return visible, points


def raycast(mesh, rays):
    """
    Given an input mesh and rays compute the distance
        Parameters:
            mesh (open3d.geometry.TriangleMesh): input mesh
            rays (numpy.ndarray[numpy.float32[number_of_points, 6]]): input rays
        Returns:
            A dictionary which contains the following keys:
                t_hit: A tensor with the distance to the first hit. The shape is {..}.
                 If there is no intersection the hit distance is inf.
                geometry_ids
                primitive_ids
                primitive_uvs
                primitive_normals
    """
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh)
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)
    return ans


def closest_points(mesh, points):
    """
    Given an input mesh and points find the closest point in the mesh surface for each point
        Parameters:
            mesh (open3d.geometry.TriangleMesh): input mesh
            points (numpy.ndarray[numpy.float32[number_of_points, 3]]): input points
        Returns:
            points (numpy.ndarray[numpy.float32[number_of_points, 3]]): output points
    """
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh)

    query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_closest_points(query_point)["points"].numpy()
    return ans

