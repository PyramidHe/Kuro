import numpy as np
import yaml
from PIL import Image


def read_img(filename, downsample = 1.0):
    img = Image.open(filename)
    img = img.resize((int(img.size[0]/downsample), int(img.size[1]/downsample)), Image.ANTIALIAS)
    # scale 0~255 to 0~1
    img_arr = np.array(img, dtype=np.float32) / 255.
    # drop alpha channel
    if img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3]
    return img_arr


def read_cam(cam_file):
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    intrinsics = np.zeros((3, 3), dtype=np.float32)

    ex_off = 1  # 1 = "extrinsics"
    in_off = ex_off + 16 + 1  # 16 = matrix of extrinsics (4*4), 1 = "intrinsics"
    depth_off = in_off + 9  # 9 = matrix of intrinsics (4*4)
    with open(cam_file, 'r') as f:
        elems = f.read().split()

        for i in range(4):
            for j in range(4):
                extrinsics[i][j] = elems[i*4+j+ex_off]
        for i in range(3):
            for j in range(3):
                intrinsics[i][j] = elems[i*3+j+in_off]

    f.close()

    return extrinsics, intrinsics


def write_cam(cam_file, extrinsic, intrinsic):
    with open(cam_file, 'w+') as f:
        string_w = 'extrinsic\n'
        for i in range(4):
            string_w = string_w + " ".join(map(str, extrinsic[i]))
            string_w = string_w + "\n"
        string_w = string_w + "\nintrinsic\n"
        for i in range(3):
            string_w = string_w + " ".join(map(str, intrinsic[i]))
            string_w = string_w + "\n"
        f.write(string_w)
    f.close()
    return

def read_points_file(points_file, mode="Train", norm_split=False):
    assert mode in ["Train", "Inference"]
    with open(points_file, 'r') as f:
        line = f.readline()
        points = []
        vecs = []
        num_cameras = len(line.split()[6:])
        nocam = []

        while line:
            split = line.split()
            points.append([float(item) for item in split[:3]])
            if mode == "Train":
                vecs.append([float(item) for item in split[3:6]])
            nocam.append([i for i, item in enumerate(split[6:]) if item == "True"])
            line = f.readline()

    points = np.array(points, dtype=np.float32)
    vecs = np.array(vecs, dtype=np.float32)
    if norm_split:
        module = np.expand_dims(np.linalg.norm(vecs, axis=1), axis=1)
        vecs = np.concatenate([vecs/module, module], axis=1)
    f.close()
    if mode == "Train":
        return points, vecs, nocam
    return points, nocam


def read_pv_file(pv_file, num=4, mode="Train"):
    assert mode in ["Train", "Inference"]
    with open(pv_file, 'r') as f:
        line = f.readline()
        points = []
        vecs = []
        scalar = []
        num_cameras = len(line.split()[6:])
        nocam = []

        while line:
            split = line.split()
            points.append([float(item) for item in split[:3]])
            vecs.append([float(item) for item in split[3:6]])
            if mode == "Train":
                scalar.append(float(split[6]))

            scores = np.array([float(item) for item in split[7:]])
            ind = np.argpartition(scores, -num)[-num:]
            nocam.append(ind)
            line = f.readline()

    points = np.array(points, dtype=np.float32)
    vecs = np.array(vecs, dtype=np.float32)
    scalar = np.array(scalar, dtype=np.float32)
    f.close()
    if mode == "Train":
        return points, vecs, scalar, nocam
    return points, vecs, nocam


def scanner_converter(yaml_file, scale_factor=1.0, test=False):
    in_matrices = []
    ex_matrices = []
    resolutions = []
    in_matrix = np.zeros((3, 3), dtype=np.float)
    ex_matrix = np.zeros((4, 4), dtype=np.float)

    with open(yaml_file, 'r') as file:
        camera_params = yaml.load(file, Loader=yaml.Loader)
        num_cam = len(camera_params["intrinsics"])
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format

    for i in range(num_cam):

        rot = camera_params["extrinsics"][i]["rotation"]["data"]
        translation = camera_params["extrinsics"][i]["translation"]["data"]
        in_matrix[0, 0] = camera_params["intrinsics"][i]["fx"]
        in_matrix[1, 1] = camera_params["intrinsics"][i]["fy"]
        in_matrix[0, 2] = camera_params["intrinsics"][i]["cx"]
        in_matrix[1, 2] = camera_params["intrinsics"][i]["cy"]
        in_matrix[2, :] = np.array([0., 0., 1.])
        resolutions.append((camera_params["intrinsics"][i]["img_width"], camera_params["intrinsics"][i]["img_height"]))
        if (test):
            in_matrix[:2, :] /= 4
        for j in range(9):
            row_index = np.floor(j / 3).astype(int)
            col_index = np.floor(j % 3).astype(int)
            ex_matrix[row_index, col_index] = rot[j]

        for j in range(3):
            ex_matrix[j, 3] = translation[j] * scale_factor

        ex_matrix[3, :] = np.array([0., 0., 0., 1.])
      # append to output
        in_matrices.append(np.copy(in_matrix))
        ex_matrices.append(np.copy(ex_matrix))

    return in_matrices, ex_matrices, resolutions


def read_matrix(filename):
    with open(filename, 'r') as f:
        line = f.readline()
        matrix = []
        split = line.split()
        num_col = len(split)
        while line:
            split = line.split()
            if not len(split) == num_col:
                raise IOError("Rows must have the same dimensions")
            row = [float(item) for item in split]
            matrix.append(row)
            line = f.readline()
    return np.array(matrix, dtype=np.float32)



