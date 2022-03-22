import numpy as np
import os
import glob
import yaml
from torch.utils.data import Dataset
from utils.file_io import read_pv_file, scanner_converter, read_img, read_cam


class PVDataset(Dataset):
    def __init__(self, datapath, listfile, num_imgs=3, mode="train", downsample=1.0):
        super().__init__()
        self.datapath = datapath
        self.num_imgs = num_imgs
        self.listfile = listfile
        self.mode = mode
        self.downsample = downsample
        self.metas = self.build_metas()



    def build_metas(self):

        with open(self.listfile, 'r') as f:
            list_yaml = yaml.load(f, Loader=yaml.Loader)
            train_list = list_yaml[self.mode]
        points_array = []
        vecs_array = []
        nocam_array = []
        scalar_array = []
        num_cam = []
        imgs_array = []
        se_index_per_sample = []
        s_index = 0
        img_folder = os.path.join(self.datapath, "Rectified")
        gt_folder = os.path.join(self.datapath, "GT")
        camera_folder = os.path.join(self.datapath, "Cameras")

        # CAMERAS
        cam_list = sorted(glob.glob(os.path.join(camera_folder, "*.txt")))
        proj_mats = []

        for cam_txt in cam_list:
            extrinsics, intrinsics = read_cam(cam_txt)
            intrinsics[:2, :] = intrinsics[:2, :]/self.downsample
            extrinsics[:3, :4] = np.matmul(intrinsics, extrinsics[:3, :4])
            proj_mats.append(extrinsics[:3, :4])

        for sample in train_list:
            gt = os.path.join(gt_folder, sample + ".npz")
            sample_img_folder = os.path.join(img_folder, sample)
            imgs_array.append(sorted(glob.glob(os.path.join(sample_img_folder, "*.png"))))
            points, vecs, scalar, nocam = read_pv_file(gt, num=self.num_imgs, mode="Train")
            points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
            vecs = np.concatenate([vecs, np.zeros((vecs.shape[0], 1), dtype=np.float32)], axis=1)
            e_index = s_index + points.shape[0] - 1
            se_index_per_sample.append((s_index, e_index))
            s_index = e_index + 1
            points_array.append(points)
            vecs_array.append(vecs)
            nocam_array.append(nocam)
            scalar_array.append(scalar)


        metas = (points_array, vecs_array, scalar_array, imgs_array, nocam_array, proj_mats, se_index_per_sample, num_cam)
        tot_len = 0
        for points in metas[0]:
            tot_len = tot_len + points.shape[0]
        print("dataset", self.mode, "points:", tot_len)
        return metas

    def __len__(self):
        tot_len = 0
        for points in self.metas[0]:
            tot_len = tot_len + points.shape[0]
        return tot_len

    def __getitem__(self, idx):
        points_array, vecs_array, scalar_array, imgs_array, nocam_array, proj_mats, se_index_per_sample, num_cam = self.metas
        sample_num = 0
        relative_idx = 0
        for i, se_index in enumerate(se_index_per_sample):
            if (idx >= se_index[0]) and (idx <= se_index[1]):
                sample_num = i
                relative_idx = idx-se_index[0]
                break
        point = np.expand_dims(points_array[sample_num][relative_idx], axis=1)
        vec = np.expand_dims(vecs_array[sample_num][relative_idx], axis=1)
        scalar = scalar_array[sample_num][relative_idx]
        confidence = 0.0
        if scalar > confidence:
            confidence = 1.0
        nocam = nocam_array[sample_num][relative_idx]
        imgs = []
        proj_mats_l = []
        for n in range(self.num_imgs):
            imgs.append(np.transpose(read_img(imgs_array[sample_num][nocam[n]], self.downsample), (2, 0, 1)))
            proj_mats_l.append(proj_mats[nocam[n]])

        return {"point": point,
                "vec": vec,
                "imgs": imgs,
                "scalar": scalar,
                "intersect": confidence,
                "proj_mats": proj_mats_l,
                 "idx": sample_num
                }
