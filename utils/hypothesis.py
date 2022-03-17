import numpy as np
import open3d as o3d
#TODO define a reference mesh for each cloud


def planecamera(intrinsics, extrinsics, resolution, depth):
    """
    Given input camera parameters, resolution and depth, return a pcl representing the plane parallel to the image plane
     at the input depth
         Parameters:
            intrinsics (numpy.ndarray[numpy.float64[3, 3]]): intrinsic matrix
            extrinsics (numpy.ndarray[numpy.float64[4, 4]]): extrinsic matrix
            resolution (tuple(int, int)): resolution of camera
            depth (float): depth of the output projected plane
        Returns:
             points (numpy.ndarray[numpy.float64[resolution[0], resolution[1]]]): points of the outptut plane
    """
    mat_cam = extrinsics.copy()
    mat_cam[:3, :4] = np.matmul(intrinsics, mat_cam[:3, :4])
    inv_mat = np.linalg.inv(mat_cam)

    # optical_center = intrinsics[:, 2]
    # opt_vec_size=2000
    # opt_vec = np.zeros((opt_vec_size, 3))
    # for r in range(opt_vec_size):
    #     ds = 1.0 + 0.01*r
    #     opt_vec[r]=np.matmul(inv_mat, np.array([optical_center[0] * ds, optical_center[1]* ds, ds, 1.0]))[:3]

    size = resolution[0]*resolution[1]
    points = np.zeros((size, 3))

    for x in range(resolution[0]):
        for y in range(resolution[1]):
            points[x*resolution[1]+y, :] = np.matmul(inv_mat, np.array([x*depth, y*depth, depth, 1.0]))[:3]
    return points


def get_spherical_hypothesis(mesh, number_of_points):
    """
    Given an input mesh and a number of points return points sampled on the surface of a sphere which contains the
    input mesh
        Parameters:
            mesh (open3d.geometry.TriangleMesh): input mesh
            number_of_points (int): number of points to sample
        Returns:
             hypothesis (open3d.geometry.PointCloud): output spherical point cloud
             center (numpy.ndarray[numpy.float64[3, 1]]): center of the output sphere
    """
    bounding_box_pts = np.asarray(mesh.get_oriented_bounding_box().get_box_points())
    center = mesh.get_center()
    rad = np.linalg.norm(center-bounding_box_pts[0])
    mesh_hypothesis = o3d.geometry.TriangleMesh.create_sphere(radius=rad, resolution=20).translate(center)
    hypothesis = mesh_hypothesis.sample_points_uniformly(number_of_points=number_of_points)
    return hypothesis, mesh_hypothesis, center


def get_random_points(mesh, number_of_points, max_range):
    """
    Given an input mesh and a number of points return points sampled on the surface of a sphere which contains the
    input mesh
        Parameters:
            mesh (open3d.geometry.TriangleMesh): input mesh
            number_of_points (int): number of points to sample
            max_range (float): parameter which regulate the max distance from the center
        Returns:
             result (numpy.ndarray[numpy.float32[number_of_points, 3]])
    """
    center = mesh.get_center()
    result = np.zeros((number_of_points, 3))
    bounding_box_pts = np.asarray(mesh.get_oriented_bounding_box().get_box_points())
    rad = np.linalg.norm(center - bounding_box_pts[0])
    if max_range < rad*1.44:
        max_range = rad*1.44

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    completed = False
    num_done = 0
    while not completed:
        rand_points = (((np.random.rand(number_of_points, 3)-0.5) * max_range) + center).astype(np.float32)
        query_points = o3d.core.Tensor(rand_points, dtype=o3d.core.Dtype.Float32)
        occupancy = scene.compute_occupancy(query_points).numpy()
        mask = occupancy < 0.5
        prev_done = num_done

        if (num_done + np.sum(mask)) > number_of_points:
            last_index = number_of_points - num_done
            num_done = number_of_points
            result[prev_done:num_done] = rand_points[mask][:last_index]
            completed = True
        else:
            num_done = num_done + np.sum(mask)
            result[prev_done:num_done] = rand_points[mask]

    return result


def get_raw_hypothesis(mesh, number_of_points, scale=2):
    """
    Given an input mesh and a number of points return points sampled on the surface (scaled input mesh) enclosing the
    input mesh
        Parameters:
            mesh (open3d.geometry.TriangleMesh): input mesh
            number_of_points (int): number of points to sample
            scale (float): scale factor, must be bigger than 1.1
        Returns:
             hypothesis (open3d.geometry.PointCloud): output point cloud
             center (numpy.ndarray[numpy.float64[3, 1]]): center of the  surface
    """
    if scale < 1.1:
        raise ValueError('scale < 1.1 not valid')
    center = mesh.get_center()
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    pcd.normalize_normals()
    points = np.asarray(pcd.points)
    ori_points = np.copy(points)
    normals = np.asarray(pcd.normals)
    for i in range(points.shape[0]):
        points[i] = points[i]+normals[i]*scale
    hypothesis = o3d.geometry.PointCloud()
    hypothesis.points = o3d.utility.Vector3dVector(points)
    return hypothesis, ori_points


def get_spherical_rays(mesh, number_of_points):
    """
    Given an input mesh and a number of points return rays starting from
    points sampled on the surface of a sphere which contains the
    input mesh and directed towards the center
        Parameters:
            mesh (open3d.geometry.TriangleMesh): input mesh
            number_of_points (int): number of points to sample
        Returns:
            rays (numpy.ndarray[numpy.float32[number_of_points, 6]]): output rays
    """
    hypothesis, hypothesis_mesh, center = get_spherical_hypothesis(mesh, number_of_points)
    points = np.asarray(hypothesis.points)
    vector = center-points
    rays = np.concatenate((points, vector), axis=1).astype(dtype=np.float32)
    return rays, points, hypothesis_mesh


def get_raw_rays(mesh, number_of_points):
    """
    Given an input mesh and a number of points return rays starting from
    points sampled on the surface which contains the
    input mesh and directed towards the center
        Parameters:
            mesh (open3d.geometry.TriangleMesh): input mesh
            number_of_points (int): number of points to sample
        Returns:
            rays (numpy.ndarray[numpy.float32[number_of_points, 6]]): output rays
    """
    hypothesis, ori_points = get_raw_hypothesis(mesh, number_of_points)
    points = np.asarray(hypothesis.points)
    vector = ori_points-points
    rays = np.concatenate((points, vector), axis=1).astype(dtype=np.float32)
    return rays


def get_mesh_rays(mesh, number_of_points):
    """
    Get rays lying on the surface
        Parameters:
            mesh (open3d.geometry.TriangleMesh): input mesh
            number_of_points (int): number of points to sample
        Returns:
            rays (numpy.ndarray[numpy.float32[number_of_points, 6]]): output rays
    """
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    pcd.normalize_normals()
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    rays = np.concatenate((points, normals), axis=1).astype(dtype=np.float32)
    return rays