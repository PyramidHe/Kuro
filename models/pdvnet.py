import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
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

    line2d0 = (line2d / 2).long()
    line2d1 = (line2d / 4).long()
    line2d2 = (line2d / 8).long()
    line2d3 = (line2d / 16).long()
    line2d4 = (line2d / 32).long()
    return line2d0, line2d1, line2d2, line2d3, line2d4


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


class Bottleneck(nn.Module):
    """Extract features"""

    def __init__(self, num_points, patch_size, channel_dim=428):
        super().__init__()

        self.b0 = Conv3BN(channel_dim, 256)
        self.b1 = Conv3BN(256, 64)
        self.b2 = Conv3BN(64, 32)
        self.b3 = Conv3BN(32, 16)
        self.b4 = Conv3BN(16, 1)
        self.pc0 = ConvBN(num_points, 8)
        self.pc1 = ConvBN(8, 4)
        self.pc2 = ConvBN(4, 1)
        self.lin = nn.Linear(patch_size*patch_size, 2)

    def forward(self, x):
        x = self.b4(self.b3(self.b2(self.b1(self.b0(x)))))
        x = torch.squeeze(x)
        x = self.pc2(self.pc1(self.pc0(x)))
        x = torch.squeeze(x)
        s0, s1, s2 = x.shape
        x = x.view(s0, s1*s2)
        x = torch.sigmoid(self.lin(x))
        return torch.squeeze(x[:, 0]), torch.squeeze(x[:, 1])


class PVNet(nn.Module):
    # TODO: if possible reduce memory footprint (remove upsampling layer?), images must be multiples of 32 (pad?)
    def __init__(self, num_points=16, patch_size=5, mode="Train"):
        super().__init__()
        self.num_points = num_points
        self.patch_size = patch_size
        self.feat_net = timm.create_model('rexnet_100', features_only=True, pretrained=True)
        # freeze the weights on feature extractor
        for param in self.feat_net.parameters():
            param.requires_grad = False

        self.bottleneck = Bottleneck(self.num_points, self.patch_size)

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
        f_downsize = 32
        width = images[0].shape[-1]
        height = images[0].shape[-2]
        hpad = f_downsize - height % f_downsize
        wpad = f_downsize - width % f_downsize
        for image in images:
            image = F.pad(image, (0, wpad, 0, hpad), "constant", 0)
            f0, f1, f2, f3, f4 = self.feat_net(image)
            features_0.append(f0)
            features_1.append(f1)
            features_2.append(f2)
            features_3.append(f3)
            features_4.append(f4)

        line2d0, line2d1, line2d2, line2d3, line2d4 = pv3d_line_index(proj_matrices, points3d, vecs3d,
                                                                              self.num_points)

        features_0 = torch.var(line_patches_extraction(patches2d(torch.stack(features_0, dim=1), f_size=self.patch_size), line2d0), dim=2)
        features_1 = torch.var(line_patches_extraction(patches2d(torch.stack(features_1, dim=1), f_size=self.patch_size), line2d1), dim=2)
        features_2 = torch.var(line_patches_extraction(patches2d(torch.stack(features_2, dim=1), f_size=self.patch_size), line2d2), dim=2)
        features_3 = torch.var(line_patches_extraction(patches2d(torch.stack(features_3, dim=1), f_size=self.patch_size), line2d3), dim=2)
        features_4 = torch.var(line_patches_extraction(patches2d(torch.stack(features_4, dim=1), f_size=self.patch_size), line2d4), dim=2)

        fused_feature = torch.cat([features_0, features_1, features_2, features_3, features_4],
                                  dim=2)  # SIZE(NP, B, C=16+38+61+128+185=428 , P, P)
        fused_feature = fused_feature.permute(1, 2, 0, 3, 4)
        vector, confidence = self.bottleneck(fused_feature)
        return vector, confidence


