import torch
#import numpy as np
from panoptic_bev.utils import plogging
logger = plogging.get_logger()
from panoptic_bev.utils.fake_ops import fake_linalg_inv
from panoptic_bev.custom.custom_inverse import custom_inverse

@torch.jit.script
def compute_M(scale, image_size, px_per_metre):
    """ image_size --> (H, W) """
    # Compute the mapping matrix from road to world (2D -> 3D)
    # px_per_metre = abs((bev_focal_length * scale) / (bev_camera_z))

    # shift --> (W, H) (Where you want the output to be placed at wrt the input dimension)
    shift = ((image_size[1] / 2 * scale), image_size[0] * scale)  # Shift so that the thing is in Bottom center

    M = torch.stack([torch.stack([1.0/px_per_metre, torch.tensor(0.0, device=scale.device), -shift[0]/px_per_metre]),
                  torch.stack([torch.tensor(0.0, device=scale.device), 1.0/px_per_metre, -shift[1]/px_per_metre]),
                  torch.tensor([0.0, 0.0, 0.0], device=scale.device),  # This must be all zeros to cancel out the effect of Z
                  torch.tensor([0.0, 0.0, 1.0], device=scale.device)], dim=0)

    return M

@torch.jit.script
def compute_intrinsic_matrix(fx, fy, px, py, img_scale):
    K = torch.stack([torch.stack([fx * img_scale, torch.tensor(0.0, device=fx.device), px * img_scale]),
                    torch.stack([torch.tensor(0.0, device=fx.device), fy * img_scale, py * img_scale]),
                    torch.tensor([0.0, 0.0, 1.0], device=fx.device)], dim=0)
    return K

@torch.jit.script
def compute_extrinsic_matrix(translation, rotation):
    # World to camera
    # theta_w2c_x = np.deg2rad(rotation[0])
    # theta_w2c_y = np.deg2rad(rotation[1])
    # theta_w2c_z = np.deg2rad(rotation[2])

    theta_w2c = torch.deg2rad(rotation)
    theta_w2c_x = theta_w2c[0]
    theta_w2c_y = theta_w2c[1]
    theta_w2c_z = theta_w2c[2]
    # logger.debug("theta_w2c: {}".format(theta_w2c))

    # R_x = np.array([[1, 0, 0],
    #                 [0, np.cos(theta_w2c_x), -np.sin(theta_w2c_x)],
    #                 [0, np.sin(theta_w2c_x), np.cos(theta_w2c_x)]], dtype=np.float)
    # R_y = np.array([[np.cos(theta_w2c_y), 0, np.sin(theta_w2c_y)],
    #                 [0, 1, 0],
    #                 [-np.sin(theta_w2c_y), 0, np.cos(theta_w2c_y)]], dtype=np.float)
    # R_z = np.array([[np.cos(theta_w2c_z), -np.sin(theta_w2c_z), 0],
    #                 [np.sin(theta_w2c_z), np.cos(theta_w2c_z), 0],
    #                 [0, 0, 1]], dtype=np.float)

    R_x = torch.stack([
        torch.stack([torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0)]),
        torch.stack([torch.tensor(0.0), torch.cos(theta_w2c_x), -torch.sin(theta_w2c_x)]),
        torch.stack([torch.tensor(0.0), torch.sin(theta_w2c_x), torch.cos(theta_w2c_x)])
    ], dim=0)
    R_y = torch.stack([
        torch.stack([torch.cos(theta_w2c_y), torch.tensor(0.0), torch.sin(theta_w2c_y)]),
        torch.stack([torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)]),
        torch.stack([-torch.sin(theta_w2c_y), torch.tensor(0.0), torch.cos(theta_w2c_y)])
    ], dim=0)
    R_z = torch.stack([
        torch.stack([torch.cos(theta_w2c_z), -torch.sin(theta_w2c_z), torch.tensor(0.0)]),
        torch.stack([torch.sin(theta_w2c_z), torch.cos(theta_w2c_z), torch.tensor(0.0)]),
        torch.tensor([0.0, 0.0, 1.0])
    ], dim=0)

    R = (R_y @ (R_x @ R_z))

    # t = -np.array(translation, dtype=np.float)
    # t_rot = np.matmul(R, np.expand_dims(t, axis=1))
    t = -1*translation
    t_rot = torch.matmul(R, t)

    # extrinsic = np.zeros((3, 4), dtype=np.float)
    # extrinsic[:3, :3] = R[:3, :3]
    # extrinsic[:, 3] = t_rot.squeeze(1)
    extrinsic = torch.zeros((3, 4), dtype=torch.float)
    extrinsic[:3, :3] = R[:3, :3]
    extrinsic[:, 3] = t_rot

    # logger.debug("translation: {}, rotataion: {}, R_x: {}, R_y: {}, R_z: {}, R: {}, t_rot: {}, extrinsc: {}".format(translation, rotation, R_x, R_y, R_z, R, t_rot, extrinsic))

    return extrinsic

@torch.jit.script
def compute_homography(intrinsic_matrix, extrinsic_matrix, M):
    P = torch.matmul(intrinsic_matrix, extrinsic_matrix)
    H = custom_inverse(P @ M)
    return H


def get_init_homography(intrinsics, extrinsics, px_per_metre, img_scale, img_size):
    # extrinsic_mat = compute_extrinsic_matrix(extrinsics['translation'], extrinsics['rotation'])
    extrinsic_mat = compute_extrinsic_matrix(extrinsics[0], extrinsics[1]).to(intrinsics.device)
    intrinsic_mat = compute_intrinsic_matrix(intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2],
                                             img_scale)
    # px_per_metre = torch.abs((bev_params['f'] * img_scale) / (bev_params['cam_z']))
    M = compute_M(img_scale, img_size, px_per_metre).to(intrinsics.device)
    H = compute_homography(intrinsic_mat, extrinsic_mat, M)
    # logger.debug("extrinsic_mat: {}, intrinsic_mat: {}, M: {}, H: {}".format(extrinsic_mat, intrinsic_mat, M, H))
    # H = torch.tensor(H.astype(np.float32))
    return H
