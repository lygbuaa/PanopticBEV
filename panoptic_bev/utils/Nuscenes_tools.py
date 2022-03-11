#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import cv2, torch
import math, os
import torch.utils.data as data
from torchvision.transforms import functional as tfn
from nuscenes.nuscenes import NuScenes
from PIL import Image
#pip install numba scipy numpy-quaternion
import quaternion
from panoptic_bev.utils import plogging
from panoptic_bev.utils.ValidMask import ValidMask
from panoptic_bev.data.misc import iss_collate_fn
from panoptic_bev.data.sampler import DistributedARBatchSampler
# plogging.init("./", "NuScenesDataLoader") #remember, in this project, plogging.init() no more than once
logger = plogging.get_logger()

# load images, intrinsics, extrinsics
class NuScenesDataLoader(object):
    VER = "v1.0-mini"  #"v1.0-trainval"
    DATAROOT = "/home/hugoliu/github/dataset/nuscenes/mini"
    NUSC = None
    SAMPLE_LEN = 0

    def __init__(self, version="v1.0-mini", dataroot="/home/hugoliu/github/dataset/nuscenes/mini"):
        self.VER = version
        self.DATAROOT = dataroot
        self.NUSC = NuScenes(version=self.VER, dataroot=self.DATAROOT, verbose=True)
        # self.NUSC.list_scenes()
        self.SAMPLE_LEN = len(self.NUSC.sample)
        # print(self.NUSC.calibrated_sensor)
    
    def get_max_idx(self):
        return self.SAMPLE_LEN

    def get_sample(self, idx):
        index = idx % self.SAMPLE_LEN
        return self.NUSC.sample[index]

    def get_sample_data(self, token):
        return self.NUSC.get("sample_data", token)

    def get_calibrated_sensor(self, token):
        return self.NUSC.get("calibrated_sensor", token)

    def get_sample_detail(self, idx=0, sensor_name="CAM_FRONT"):
        sample = self.get_sample(idx)
        sample_token = sample["data"][sensor_name]
        sample_data = self.get_sample_data(sample_token)
        filename = os.path.join(self.DATAROOT, sample_data["filename"])
        calib_token = sample_data['calibrated_sensor_token']
        calib = self.get_calibrated_sensor(calib_token)
        intrinsic = calib["camera_intrinsic"]
        translation = calib["translation"]
        rotation = calib["rotation"]
        q_rot = np.quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
        eulers = quaternion.as_euler_angles(q_rot)*180/math.pi # return ndarray
        eulers = [angle if angle>0 else 360+angle for angle in eulers] # normalize to 0~360
        return filename, intrinsic, translation, rotation, eulers

class BEVTransformV2(object):
    def __init__(self,
                 shortest_size,
                 longest_max_size,
                 rgb_mean=None,
                 rgb_std=None,
                 scale=None,
                 front_resize=None):
        self.shortest_size = shortest_size
        self.longest_max_size = longest_max_size
        
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.scale = scale
        if self.scale != None:
            self.front_resize = (int(front_resize[0]*self.scale), int(front_resize[1]*self.scale))
        else:
            self.front_resize = front_resize

    def _resize(self, img, mode):
        if img is not None:
            # Resize the image
            out_img_w, out_img_h = self.front_resize[1], self.front_resize[0]
            img = [rgb.resize((out_img_w, out_img_h), mode) for rgb in img]

        return img

    def _normalize_image(self, img):
        if img is not None:
            if (self.rgb_mean is not None) and (self.rgb_std is not None):
                img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
                img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img

    def __call__(self, img_list, intrinsics_list):
        # Resize the RGB image and the front mask to the given dimension
        if self.front_resize:
            img_list = self._resize(img_list, Image.BILINEAR)
        # Image transformations
        img_list = [tfn.to_tensor(rgb) for rgb in img_list]
        img_list = [self._normalize_image(rgb) for rgb in img_list]
        # Concatenate the images to make it easier to process downstream
        # don't concatenate, just keep it as list
        # img_list = torch.cat(img_list, dim=0)

        # Adjust calib and wrap in np.array
        intrin_scale_list = []
        for calib in intrinsics_list:
            calib = np.array(calib, dtype=np.float32)
            if len(calib.shape) == 3:
                calib[:, 0, 0] *= float(self.front_resize[1]) / self.longest_max_size
                calib[:, 1, 1] *= float(self.front_resize[0]) / self.shortest_size
                calib[:, 0, 2] *= float(self.front_resize[1]) / self.longest_max_size
                calib[:, 1, 2] *= float(self.front_resize[0]) / self.shortest_size
            else:
                calib[0, 0] *= float(self.front_resize[1]) / self.longest_max_size
                calib[1, 1] *= float(self.front_resize[0]) / self.shortest_size
                calib[0, 2] *= float(self.front_resize[1]) / self.longest_max_size
                calib[1, 2] *= float(self.front_resize[0]) / self.shortest_size
            calib = torch.from_numpy(calib)
            intrin_scale_list.append(calib)

        return img_list, intrin_scale_list

class BEVNuScenesDatasetV2(data.Dataset):
    transform = None
    dataloader = None
    idx = 0
    nuscenes_root_dir = None
    rgb_cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT", "CAM_FRONT_RIGHT"]
    valid_masks = None
    bev_size = [896, 768]
    fov_borders = {}

    def __init__(self, nuscenes_version, nuscenes_root_dir, transform, bev_size, scale = None):
        super(BEVNuScenesDatasetV2, self).__init__()
        self.nuscenes_root_dir = nuscenes_root_dir
        self.transform = transform
        self.dataloader = NuScenesDataLoader(version=nuscenes_version, dataroot=nuscenes_root_dir)
        self.idx = 0
        
        if scale != None:
            self.bev_size = (int(bev_size[0]*scale), int(bev_size[1]*scale))
        else:
            self.bev_size = bev_size
        self.init_valid_mask()

    def init_valid_mask(self):
        self.valid_masks = {}
        self.fov_borders = {}
        ValidMsk = ValidMask()
        H = self.bev_size[0]
        W = self.bev_size[1]
        for cam_name in self.rgb_cameras:
            if cam_name == "CAM_FRONT":
                self.valid_masks[cam_name] = ValidMsk.generate(h=H, w=W, fov_l=30, fov_r=30)
            elif cam_name == "CAM_FRONT_LEFT":
                self.valid_masks[cam_name] = ValidMsk.generate(h=H, w=W, fov_l=30, fov_r=24.8)
            elif cam_name == "CAM_BACK_LEFT": 
                self.valid_masks[cam_name] = ValidMsk.generate(h=H, w=W, fov_l=30, fov_r=23.5)
            elif cam_name == "CAM_BACK": 
                self.valid_masks[cam_name] = ValidMsk.generate(h=H, w=W, fov_l=42, fov_r=42)
            elif cam_name == "CAM_BACK_RIGHT": 
                self.valid_masks[cam_name] = ValidMsk.generate(h=H, w=W, fov_l=30, fov_r=30)
            elif cam_name == "CAM_FRONT_RIGHT": 
                self.valid_masks[cam_name] = ValidMsk.generate(h=H, w=W, fov_l=26.7, fov_r=24.4)

    def calc_valid_mask(self, cam_name, ex_rot):
        ValidMsk = ValidMask()
        H = self.bev_size[0]
        W = self.bev_size[1]
        MAGIC_MARGIN = 3.0
        if cam_name == "CAM_FRONT":
            self.fov_borders[cam_name + "_L"] = 30.0
            self.fov_borders[cam_name + "_R"] = 360.0 - 30.0
            return ValidMsk.generate(h=H, w=W, fov_l=30+MAGIC_MARGIN, fov_r=30+MAGIC_MARGIN)
        elif cam_name == "CAM_FRONT_LEFT":
            self.fov_borders[cam_name + "_L"] = ex_rot[0] + 30.0
            self.fov_borders[cam_name + "_R"] = self.fov_borders["CAM_FRONT_L"]
            return ValidMsk.generate(h=H, w=W, fov_l=30+MAGIC_MARGIN, fov_r=(ex_rot[0]-self.fov_borders["CAM_FRONT_LEFT_R"]+MAGIC_MARGIN))
        elif cam_name == "CAM_BACK_LEFT":
            self.fov_borders[cam_name + "_L"] = ex_rot[0] + 30.0
            self.fov_borders[cam_name + "_R"] = self.fov_borders["CAM_FRONT_LEFT_L"]
            return ValidMsk.generate(h=H, w=W, fov_l=30+MAGIC_MARGIN, fov_r=(ex_rot[0]-self.fov_borders["CAM_BACK_LEFT_R"]+MAGIC_MARGIN))
        elif cam_name == "CAM_BACK":
            self.fov_borders[cam_name + "_L"] = ex_rot[0] + 42.0
            self.fov_borders[cam_name + "_R"] = self.fov_borders["CAM_BACK_LEFT_L"]
            return ValidMsk.generate(h=H, w=W, fov_l=42+MAGIC_MARGIN, fov_r=ex_rot[0]-self.fov_borders["CAM_BACK_R"]+MAGIC_MARGIN)
        elif cam_name == "CAM_BACK_RIGHT":
            self.fov_borders[cam_name + "_L"] = ex_rot[0] + 30.0
            self.fov_borders[cam_name + "_R"] = self.fov_borders["CAM_BACK_L"]
            return ValidMsk.generate(h=H, w=W, fov_l=30+MAGIC_MARGIN, fov_r=ex_rot[0]-self.fov_borders["CAM_BACK_RIGHT_R"]+MAGIC_MARGIN)
        elif cam_name == "CAM_FRONT_RIGHT":
            self.fov_borders[cam_name + "_L"] = self.fov_borders["CAM_FRONT_R"]
            self.fov_borders[cam_name + "_R"] = self.fov_borders["CAM_BACK_RIGHT_L"]
            return ValidMsk.generate(h=H, w=W, fov_l=self.fov_borders["CAM_FRONT_RIGHT_L"]-ex_rot[0]+MAGIC_MARGIN, fov_r=ex_rot[0]-self.fov_borders["CAM_FRONT_RIGHT_R"]+MAGIC_MARGIN)

    def load_images(self, item_idx):
        # Get the RGB file names
        img_list = []
        intrinsics_list = []
        extrinsics_list = []
        valid_msk_list = []
        _, _, front_camera_trans, _, front_camera_eulers = self.dataloader.get_sample_detail(idx=item_idx, sensor_name="CAM_FRONT")
        for cam_name in self.rgb_cameras:
            filename, intrinsic, translation, rotation, eulers = self.dataloader.get_sample_detail(idx=item_idx, sensor_name=cam_name)
            if not os.path.exists(filename):
                raise IOError("image {} not found!".format(filename))
            img_list.append(Image.open(filename).convert(mode="RGB"))
            intrinsics_list.append(intrinsic)
            extrinsic = {}
            extrinsic["trans"] = [t-t0 for t,t0 in zip(translation, front_camera_trans)]
            extrinsic["rot"] = []
            # rotation angle relative to front_camera, rot = (eulers - front_camera_eulers).tolist() 
            for e, ef in zip(eulers, front_camera_eulers):
                r = e - ef
                if r < 0:
                    r += 360
                elif r > 360:
                    r -= 360
                extrinsic["rot"].append(r)
            extrinsics_list.append(torch.tensor([extrinsic["trans"], extrinsic["rot"]], dtype=torch.float))
            valid_mask = self.calc_valid_mask(cam_name, extrinsic["rot"])
            valid_msk_list.append(torch.from_numpy(valid_mask))

        return img_list, intrinsics_list, extrinsics_list, valid_msk_list

    @property
    def dataset_name(self):
        return "nuScenes"

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [self.bev_size for i in range(self.__len__())]

    @property
    def categories(self):
        """Category names"""
        return ['flat.driveable_surface', 'flat.sidewalk', 'static.manmade', 'static.vegetation', 'flat.terrain', 'occlusion', 'human.pedestrian.adult', 'vehicle.car', 'vehicle.truck', 'vehicle.motorcycle']

    @property
    def num_categories(self):
        """Number of categories"""
        return len(self.categories)

    @property
    def num_stuff(self):
        """Number of "stuff" categories"""
        return 6

    @property
    def num_thing(self):
        """Number of "thing" categories"""
        return 4

    def __len__(self):
        return self.dataloader.get_max_idx()

    # when do_loss=False, we only need 4 items: img, calib, extrinsics, valid_msk
    # img, bev_msk=None, front_msk=None, weights_msk=None, cat=None, iscrowd=None, bbx=None, calib, extrinsics, valid_msk
    def __getitem__(self, index):
        # step1: load image*6, intrinsics*6, extrinsics*6
        imgs, intrinsics, extrinsics_list, valid_msk_list = self.load_images(index)
        size = (imgs[0].size[1], imgs[0].size[0])
        img_list, intrinsics_list, = self.transform(img_list=imgs, intrinsics_list=intrinsics)

        # Close the files
        for i in imgs:
            i.close()

        # img: list of torch.tensor, calib: list of torch.tensor, extrinsics: list of dict, valid_msk: list of np.array
        rec = dict(img=torch.stack(img_list, dim=0), calib=torch.stack(intrinsics_list, dim=0), 
                extrinsics=torch.stack(extrinsics_list, dim=0), valid_msk=torch.stack(valid_msk_list, dim=0),
                bev_msk=None, front_msk=None, weights_msk=None, cat=None, iscrowd=None, bbx=None, idx=index, size=size)
        # logger.debug("load data-{}, img: {}, intrinsics: {}, extrinsics: {}, valid_msk: {}".format(index, len(img_list), intrinsics_list, extrinsics_list, len(valid_msk_list)))
        logger.debug("getitem-{}, img: {}, intrin: {}, extrin: {}, msk: {}".format(index, rec["img"].shape, rec["calib"].shape, rec["extrinsics"].shape, rec["valid_msk"].shape))
        return rec

def print_cameras(index=0):
    dataloader = NuScenesDataLoader(version="v1.0-mini", dataroot="/home/hugoliu/github/dataset/nuscenes/mini")
    filename, intrinsic, translation, rotation, eulers = dataloader.get_sample_detail(idx=index, sensor_name="CAM_FRONT")
    logger.info("sensor {} filename: {}, intrinsic: {}, translation: {}, rotation: {}, eulers: {}".format("CAM_FRONT", filename, intrinsic, translation, rotation, eulers))
    filename, intrinsic, translation, rotation, eulers = dataloader.get_sample_detail(idx=index, sensor_name="CAM_FRONT_LEFT")
    logger.info("sensor {} filename: {}, intrinsic: {}, translation: {}, rotation: {}, eulers: {}".format("CAM_FRONT_LEFT", filename, intrinsic, translation, rotation, eulers))
    filename, intrinsic, translation, rotation, eulers = dataloader.get_sample_detail(idx=index, sensor_name="CAM_BACK_LEFT")
    logger.info("sensor {} filename: {}, intrinsic: {}, translation: {}, rotation: {}, eulers: {}".format("CAM_BACK_LEFT", filename, intrinsic, translation, rotation, eulers))
    filename, intrinsic, translation, rotation, eulers = dataloader.get_sample_detail(idx=index, sensor_name="CAM_BACK")
    logger.info("sensor {} filename: {}, intrinsic: {}, translation: {}, rotation: {}, eulers: {}".format("CAM_BACK", filename, intrinsic, translation, rotation, eulers))
    filename, intrinsic, translation, rotation, eulers = dataloader.get_sample_detail(idx=index, sensor_name="CAM_BACK_RIGHT")
    logger.info("sensor {} filename: {}, intrinsic: {}, translation: {}, rotation: {}, eulers: {}".format("CAM_BACK_RIGHT", filename, intrinsic, translation, rotation, eulers))
    filename, intrinsic, translation, rotation, eulers = dataloader.get_sample_detail(idx=index, sensor_name="CAM_FRONT_RIGHT")
    logger.info("sensor {} filename: {}, intrinsic: {}, translation: {}, rotation: {}, eulers: {}".format("CAM_FRONT_RIGHT", filename, intrinsic, translation, rotation, eulers))

def test_dataset():
    tfv2 = BEVTransformV2(shortest_size=900,
                          longest_max_size=1600,
                          rgb_mean=(0.485, 0.456, 0.406),
                          rgb_std=(0.229, 0.224, 0.225),
                          front_resize=(448, 768))
    datasetv2 = BEVNuScenesDatasetV2(nuscenes_version="v1.0-mini", nuscenes_root_dir="/home/hugoliu/github/dataset/nuscenes/mini", transform=tfv2, bev_size=[896, 768])
    logger.info("load dataset len: {}".format(len(datasetv2)))
    test_sampler = DistributedARBatchSampler(datasetv2, batch_size=1, num_replicas=1, rank=0, drop_last=False)
    test_dl = torch.utils.data.DataLoader(datasetv2, batch_sampler=test_sampler, batch_size=1, collate_fn=iss_collate_fn, pin_memory=True, num_workers=1)
    for it, sample in enumerate(test_dl):
        logger.debug("enumerate-{}, img: {}, intrin: {}, extrin: {}, msk: {}".format(it, sample["img"][0].shape, sample["calib"][0].shape, sample["extrinsics"][0].shape, sample["valid_msk"][0].shape))
        if it > 2:
            # after iss_collate_fn, every param turns into PackedSequence
            for img, intrin, extrin, msk in zip(sample["img"][0], sample["calib"][0], sample["extrinsics"][0], sample["valid_msk"][0]):
                logger.debug("camera-{}: img: {}, intrin: {}, extrin: {}, msk: {}".format(it, img.shape, intrin.shape, extrin.shape, msk.shape))
            break


if __name__ == '__main__':
    print_cameras(index=100)
    # test_dataset()
