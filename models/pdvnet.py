import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import ConvBN, Conv3BN, UpConv

# B = batch size
# C = channel size
# H = height
# W = width
# P = patch size
# N = number of images
# NP = number of points in the line

def pv3d_line_index(proj_matrices, points3d, vecs3d, num):
    """
    For each couple (input 3D points and 3D direction vector) extract num points lying in the line projected
    in the image, each for the N input camera matrices
        Parameters:
            proj_matrices (list of N tensor of size (B, 3, 4)): camera matrices
            points3d (tensor of size (B, 4, 1)): input 3D points
            vecs3d (tensor of size (B, 4, 1)): input 3D vectors
            num (int): number of points to sample in the line projected in the image
            resolution (tuple(int, int)): mage resolution
        Returns:
            line2dn for n = 0, ..., L (L+1 tensors of size (NP=num, B, N, 2))
    """

    steps = torch.arange(num, dtype=torch.float32, device=points3d.device) / (num - 1)
    line3d = points3d[None] + steps * vecs3d[None]
    line3d = torch.permute(line3d, (3, 1, 2, 0))
    line2d = []
    for proj_matrix in proj_matrices:
        line2d.append(torch.matmul(proj_matrix, line3d))
    # the output starts at 'start' and increments until 'stop' in each dimension
    line2d = torch.stack(line2d, dim=1).squeeze()
    line2d = torch.permute(line2d, (0, 2, 1, 3))
    line2d = line2d / (line2d[:, :, :, 2].unsqueeze(-1))
    line2d = line2d[:, :, :, :2]
    line2d0 = line2d.long()
    line2d1 = (line2d / 2).long()
    line2d2 = (line2d / 4).long()
    line2d3 = (line2d / 8).long()
    line2d4 = (line2d / 16).long()
    line2d5 = (line2d / 32).long()
    return line2d0, line2d1, line2d2, line2d3, line2d4, line2d5



def patches2d(in_tensor, f_size=3):
    """
    Create patches for all the values from the last two dimensions:
        INPUT SIZE: (B, ..., C, H, W)
        OUTPUT SIZE: (B, ..., C, H, W, P, P)
    """
    assert f_size % 2 == 1, "Size must be odd!"
    in_tensor = F.pad(in_tensor, (f_size // 2, f_size // 2, f_size // 2, f_size // 2))
    in_tensor = in_tensor.unfold(-2, f_size, 1).unfold(-2, f_size, 1)
    return in_tensor


def line_patches_extraction(patches_tensor, indices):
    """
    Extract patches P*P given a set of indices:
        patches_tensor SIZE: (B, ..., C, H, W, P, P)
        indices SIZE: (NP, B, ..., 2) xs = indices[..., 0] ys = indices[..., 1]
        OUTPUT SIZE: (NP, B, ..., C, P, P)
    """
    dim_params = patches_tensor.shape
    num_dim = len(dim_params)+1
    c_dim = patches_tensor.shape[-5]
    h_dim = patches_tensor.shape[-4]
    patch_dim = patches_tensor.shape[-2]
    num_points = indices.shape[0]


    rep_x = [1] * num_dim
    rep_x[-1] = patch_dim
    rep_x[-2] = patch_dim
    rep_x[-5] = c_dim
    rep_x[-4] = h_dim

    rep_y = [1] * (num_dim - 1)
    rep_y[-1] = patch_dim
    rep_y[-2] = patch_dim
    rep_y[-4] = c_dim

    xs = indices[..., 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(rep_x)
    ys = indices[..., 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(rep_y)

    patches_ex = []
    for i in range(num_points):
        patches_ex_x = torch.gather(patches_tensor, -3, xs[i]).squeeze()
        patches_ex.append(torch.gather(patches_ex_x, -3, ys[i]).squeeze())

    patches_ex = torch.stack(patches_ex, dim=0)
    return patches_ex

class FeatureNet(nn.Module):
    """Extract features"""

    def __init__(self):
        super().__init__()
        self.b0 = ConvBN(3, 8)  # (B, C=3, H, W) => (B, C=8, H, W)

        self.b1 = ConvBN(8, 16, 2)  # (B, C=8, H, W) => (B, C=16, H/2, W/2)
        self.b1p = ConvBN(16, 16)  # (B, C=16, H/2, W/2) => (B, C=16, H/2, W/2)

        self.b2 = ConvBN(16, 32, 2)  # (B, C=16, H/2, W/2) => (B, C=32, H/4, W/4)
        self.b2p = ConvBN(32, 32)  # (B, C=32, H/4, W/4) => (B, C=32, H/4, W/4)

        self.b3 = ConvBN(32, 64, 2)  # (B, C=32, H/4, W/4) => (B, C=64, H/8, W/8)
        self.b3p = ConvBN(64, 64)  # (B, C=64, H/8, W/8) => (B, C=64, H/8, W/8)

        self.b4 = ConvBN(64, 128, 2)  # (B, C=64, H/8, W/8) => (B, C=128, H/16, W/16)
        self.b4p = ConvBN(128, 128)  # (B, C=128, H/16, W/16) => (B, C=128, H/16, W/16)

        self.b5 = ConvBN(128, 256, 2)  # (B, C=128, H/16, W/16) => (B, C=256, H/32, W/32)
        self.l5 = ConvBN(256, 256)  # (B, C=256, H/32, W/32) => (B, C=256, H/32, W/32)

        self.l4 = UpConv(256, 64)  # (B, C=256, H/32, W/32) => (B, C=128, H/16, W/16)
        self.l3 = UpConv(192, 32)  # (B, C=128, H/16, W/16) => (B, C=64, H/8, W/8)
        self.l2 = UpConv(96, 16)  # (B, C=64, H/8, W/8) => (B, C=32, H/4, W/4)
        self.l1 = UpConv(48, 8)  # (B, C=32, H/4, W/4) => (B, C=16, H/2, W/2)
        self.l0 = UpConv(24, 4)  # (B, C=16, H/2, W/2) => (B, C=8, H, W)

    def forward(self, x):
        x0d = self.b0(x)
        x1d = self.b1p(self.b1(x0d))
        x2d = self.b2p(self.b2(x1d))
        x3d = self.b3p(self.b3(x2d))
        x4d = self.b4p(self.b4(x3d))
        x5 = self.l5(self.b5(x4d))
        x4 = self.l4(x5)
        x3 = self.l3(torch.cat([x4, x4d], 1))
        x2 = self.l2(torch.cat([x3, x3d], 1))
        x1 = self.l1(torch.cat([x2, x2d], 1))
        x0 = self.l0(torch.cat([x1, x1d], 1))

        return x0, x1, x2, x3, x4, x5


# class Bottleneck(nn.Module):
#     """Extract features"""
#
#     def __init__(self, channel_dim=380):
#         super().__init__()
#
#         self.b0 = ConvBN(channel_dim, 256)
#         self.b1 = ConvBN(256, 128)
#         self.b2 = ConvBN(128, 32)
#         self.b3 = ConvBN(32, 8)
#         self.lin = nn.Linear(8, 3)
#
#     def forward(self, x):
#         x = self.b3(self.b2(self.b1(self.b0(x))))
#         x = torch.mean(torch.flatten(x, -2), -1)
#         x = self.lin(x)
#         return x


class Bottleneck(nn.Module):
    """Extract features"""

    def __init__(self, num_points, channel_dim=380):
        super().__init__()

        self.b0 = Conv3BN(channel_dim, 256)
        self.b1 = Conv3BN(256, 64)
        self.b2 = Conv3BN(64, 32)
        self.b3 = Conv3BN(32, 16)
        self.b4 = Conv3BN(16, 1)
        self.lin = nn.Linear(num_points, 1)

    def forward(self, x):
        x = self.b4(self.b3(self.b2(self.b1(self.b0(x)))))
        x = torch.squeeze(x)
        s0, s1, s2, s3 = x.shape
        x = x.view(s0, s1, s2*s3)
        x = torch.mean(x, dim=-1)
        x = torch.sigmoid(self.lin(x))
        return torch.squeeze(x)


class Bottleneck_(nn.Module):
    """Extract features"""

    def __init__(self, num_points, channel_dim=380):
        super().__init__()

        self.b0 = ConvBN(channel_dim * num_points, 512)
        self.b1 = ConvBN(512, 256)
        self.b2 = ConvBN(256, 64)
        self.b3 = ConvBN(64, 32)
        self.b4 = ConvBN(32, 16)
        self.b5 = ConvBN(16, 1)


    def forward(self, x):
        s0, s1, s2, s3, s4 = x.shape
        x = x.reshape(s0, s1*s2, s3, s4)
        x = self.b5(self.b4(self.b3(self.b2(self.b1(self.b0(x))))))
        x = torch.squeeze(x)
        x = torch.mean(x, dim=[-1, -2])
        x = torch.sigmoid(x)
        return x


class PVNet(nn.Module):
    # TODO: if possible reduce memory footprint (remove upsampling layer?), images must be multiples of 32 (pad?)
    def __init__(self, num_points=16, mode="Train"):
        super().__init__()
        self.feat_net = FeatureNet()
        self.bottleneck = Bottleneck_(num_points)
        self.num_points = num_points
        if mode not in ["Train", "Inference"]:
            raise ValueError("mode must \"Train\" or \"Inference\", different values are not accepted")
        self.mode = mode

    def forward(self, images, proj_matrices, points3d, vecs3d):
        """
            images SIZE: N * (B, C=3, H (H=32*uy), W (W=32*ux))
            proj_matrices SIZE: N * (B, 4, 3)
            points3d SIZE: (B, 4, 1)
        """

        features_0 = []
        features_1 = []
        features_2 = []
        features_3 = []
        features_4 = []
        features_5 = []
        f_downsize = 32
        width = images[0].shape[-1]
        height = images[0].shape[-2]
        hpad = f_downsize - height % f_downsize
        wpad = f_downsize - width % f_downsize
        for image in images:
            # TO DO remove hardcoded pad

            image = F.pad(image, (0, wpad, 0, hpad), "constant", 0)

            f0, f1, f2, f3, f4, f5 = self.feat_net(image)
            features_0.append(f0)
            features_1.append(f1)
            features_2.append(f2)
            features_3.append(f3)
            features_4.append(f4)
            features_5.append(f5)

        line2d0, line2d1, line2d2, line2d3, line2d4, line2d5 = pv3d_line_index(proj_matrices, points3d, vecs3d,
                                                                              self.num_points)
        # TODO remove variance: in this case it doesn't make sense
        features_0 = torch.var(line_patches_extraction(patches2d(torch.stack(features_0, dim=1), f_size=5), line2d0), dim=2)
        features_1 = torch.var(line_patches_extraction(patches2d(torch.stack(features_1, dim=1), f_size=5), line2d1), dim=2)
        features_2 = torch.var(line_patches_extraction(patches2d(torch.stack(features_2, dim=1), f_size=5), line2d2), dim=2)
        features_3 = torch.var(line_patches_extraction(patches2d(torch.stack(features_3, dim=1), f_size=5), line2d3), dim=2)
        features_4 = torch.var(line_patches_extraction(patches2d(torch.stack(features_4, dim=1), f_size=5), line2d4), dim=2)
        features_5 = torch.var(line_patches_extraction(patches2d(torch.stack(features_5, dim=1), f_size=5), line2d5), dim=2)

        fused_feature = torch.cat([features_0, features_1, features_2, features_3, features_4, features_5],
                                  dim=2)  # SIZE(NP, B, C=4+16+32+64+128+256=500# , P, P)
        fused_feature = fused_feature.permute(1, 2, 0, 3, 4)
        vector = self.bottleneck(fused_feature)
        # TODO change confidence metric
        if self.mode == "Inference":
            pred_points3d = points3d
            confidence = 0.0
            # with torch.no_grad():
            #     pred_points3d = points3d
            #     pred_points3d[:, :3] = pred_points3d[:, :3] + torch.unsqueeze(vector, -1)
            #     pred_points2d0, pred_points2d1, pred_points2d2, pred_points2d3, pred_points2d4, pred_points2d5 = points3d_index(proj_matrices, pred_points3d)
            #
            #
            #     if torch.any(pred_points2d0 < 0) or torch.any(pred_points2d0[:, :, 0] > images[0].shape[3]) or torch.any(pred_points2d0[:, :, 1] > images[0].shape[2]):
            #         confidence = 0.0
            #
            #     else:
            #         confidence = 0.0
            #         # pred_features_0 = torch.var(
            #         #     patches_extraction(patches2d(torch.stack(features_0, dim=1), f_size=5), pred_points2d0), dim=1)
            #         # pred_features_1 = torch.var(
            #         #     patches_extraction(patches2d(torch.stack(features_1, dim=1), f_size=5), pred_points2d1), dim=1)
            #         # pred_features_2 = torch.var(
            #         #     patches_extraction(patches2d(torch.stack(features_2, dim=1), f_size=5), pred_points2d2), dim=1)
            #         # pred_features_3 = torch.var(
            #         #     patches_extraction(patches2d(torch.stack(features_3, dim=1), f_size=5), pred_points2d3), dim=1)
            #         # pred_features_4 = torch.var(
            #         #
            #         #     patches_extraction(patches2d(torch.stack(features_4, dim=1), f_size=5), pred_points2d4), dim=1)
            #         # pred_features_5 = torch.var(
            #         #     patches_extraction(patches2d(torch.stack(features_5, dim=1), f_size=5), points2d5), dim=1)
            #         #
            #         # pred_fused_feature = torch.cat(
            #         #     [pred_features_0, pred_features_1, pred_features_2, pred_features_3, pred_features_4,
            #         #      pred_features_5], dim=1)
            #         #
            #         # confidence_err = torch.tensor.view(pred_fused_feature, -1).mean(1)
            #         # confidence_err = confidence_err*confidence_err
            #         # confidence = 1.0 - torch.sigmoid(confidence_err)
            return vector, confidence
        return vector


import numpy as np
from utils.file_io import read_points_file, scanner_converter, read_img
import glob
import os
import matplotlib.pyplot as plt
if __name__=="__main__":
    imgs = []
    se_index_per_sample = []
    s_index = 0
    proj_mats = []

    downsample = 2.0
    cam_file = "/home/flexsight/FlexSight/mvsonet/data/dataset/statue/scanner_config.yml"
    img_folder = "/home/flexsight/FlexSight/mvsonet/data/dataset/statue/imgs"
    point_file = "/home/flexsight/FlexSight/mvsonet/data/dataset/statue/points_vec.txt"
    output_file = "/home/flexsight/FlexSight/mvsonet/data/dataset/statue/result.txt"
    imgs_array = sorted(glob.glob(os.path.join(img_folder, "*.png")))
    points, vecs, nocam = read_points_file(point_file, mode="Train")


    true_points = points + vecs
    true_points = np.concatenate([true_points, np.ones((true_points.shape[0], 1))], axis=1)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    vecs = np.concatenate([vecs, np.zeros((vecs.shape[0], 1))], axis=1)

    for img_name in imgs_array:
        img = read_img(img_name, downsample)
        imgs.append(np.transpose(img, (2, 0, 1)))

    in_matrices, ex_matrices, _ = scanner_converter(cam_file, scale_factor=1.0, test=False)
    for i, ex_mat in enumerate(ex_matrices):
        proj_mat = ex_mat.copy()
        in_matrix = in_matrices[i]
        in_matrix[:2, :] = in_matrix[:2, :] / downsample
        proj_mat[:3, :4] = np.matmul(in_matrices[i], proj_mat[:3, :4])
        proj_mats.append(proj_mat[:3, :4])



    batch_size = 2
    num_imgs = 4
    num_points = points.shape[0]

    num_it = num_points//batch_size
    batch_size_li = num_points%batch_size


    #model = torch.load("/home/flexsight/Downloads/model_000001.ckpt")
    for i in range(num_it):
        print(i)
        bimgs = []
        bpmat = []
        pts = np.copy(points[i*batch_size:(i+1)*(batch_size)])
        pts = np.expand_dims(pts, axis=-1)

        vecs_ = np.copy(vecs[i * batch_size:(i + 1) * (batch_size)])
        vecs_ = np.expand_dims(vecs_, axis=-1)

        tpts = np.copy(true_points[i * batch_size:(i + 1) * (batch_size)])
        tpts = np.expand_dims(tpts, axis=-1)

        indices = nocam[i * batch_size:(i + 1) * (batch_size)]
        for n in range(num_imgs):
            nindices = [el[n] for el in indices]
            simg = []
            spmat = []
            for j in range(batch_size):
                simg.append(np.array(imgs[nindices[j]]))
                spmat.append(proj_mats[nindices[j]])

            print("DONE")
            spmat = torch.tensor(np.array(spmat), dtype=torch.float32)
            pts = torch.tensor(pts, dtype=torch.float32)
            vecs_ = torch.tensor(vecs_, dtype=torch.float32)
            #rr=pts+vecs_
            tpts = torch.tensor(tpts, dtype=torch.float32)
            bimgs.append(torch.tensor(simg, dtype=torch.float32))
            print(torch.tensor(simg).shape)
            bpmat.append(spmat)


        feat_net = FeatureNet()
        features_0 = []
        features_1 = []
        features_2 = []
        features_3 = []
        features_4 = []
        features_5 = []
        f_downsize = 32
        width = bimgs[0].shape[-1]
        height = bimgs[0].shape[-2]
        hpad = f_downsize - height % f_downsize
        wpad = f_downsize - width % f_downsize
        for image in bimgs:
            # TO DO remove hardcoded pad

            image = F.pad(image, (0, wpad, 0, hpad), "constant", 0)

            f0, f1, f2, f3, f4, f5 = feat_net(image)
            features_0.append(f0)
            features_1.append(f1)
            features_2.append(f2)
            features_3.append(f3)
            features_4.append(f4)
            features_5.append(f5)
        resolution = (features_0[0].shape[3], features_0[0].shape[2])
        line2d0, line2d1, line2d2, line2d3, line2d4, line2d5 = pv3d_line_index(bpmat, pts, vecs_, 30, resolution)
        #al = pv3d_line_index(bpmat, pts, tpts, 456, (1296, 972))

        features_0 = torch.var(line_patches_extraction(patches2d(torch.stack(features_0, dim=1), f_size=5), line2d0),
                               dim=2)
        features_1 = torch.var(line_patches_extraction(patches2d(torch.stack(features_1, dim=1), f_size=5), line2d1),
                               dim=2)
        features_2 = torch.var(line_patches_extraction(patches2d(torch.stack(features_2, dim=1), f_size=5), line2d2),
                               dim=2)
        features_3 = torch.var(line_patches_extraction(patches2d(torch.stack(features_3, dim=1), f_size=5), line2d3),
                               dim=2)
        features_4 = torch.var(line_patches_extraction(patches2d(torch.stack(features_4, dim=1), f_size=5), line2d4),
                               dim=2)
        features_5 = torch.var(line_patches_extraction(patches2d(torch.stack(features_5, dim=1), f_size=5), line2d5),
                               dim=2)

        fused_feature = torch.cat([features_0, features_1, features_2, features_3, features_4, features_5],
                                  dim=2)  # SIZE(NP, B, C=4+16+32+64+128+256=500# , P, P)
        fused_feature = fused_feature.permute(1, 2, 0, 3, 4)

        bottleneck = Bottleneck(30)
        vector = bottleneck(fused_feature)
        al = pv3d_line_index(bpmat, pts, vecs_, 456, (1296, 972))[0]
        for n in range(num_imgs):
            print("")
            #pointse = al.permute(0, 3, 2, 1)
            #pointse = al[:, 0, n, :]
            pointse = al[:, 0, n, :]
            #pointse = al[:, :, n, 0]


            #alle = all[0, n, :]
            imago = bimgs[n]
            imago = imago[0]
            for i in range(pointse.shape[0]):
                imago[:, pointse[i, 1], pointse[i, 0]] = torch.tensor([1.0, 0.0, 1.0])
                #P = imago[:, alle[1]-10:alle[1]+10, alle[0]-10:alle[0]+10]
            g_patch = torch.unsqueeze(torch.unsqueeze(torch.tensor([0.0, 1.0, 0.0]), dim=-1), dim=-1)
            g_patch = g_patch.repeat(1, 20, 20)
            #imago[:, alla[1] - 10:alla[1] + 10, alla[0] - 10:alla[0] + 10] = g_patch
            #imago[:, alle[0] - 10:alle[0] + 10, alle[1] - 10:alle[1] + 10] = g_patch
            print()

            # okll=c[0, n, :, :, :]
            # okll[:, x, y] = torch.tensor([1.0, 0.0, 0.0])

            plt.imshow(imago.permute(1, 2, 0))
            plt.show()