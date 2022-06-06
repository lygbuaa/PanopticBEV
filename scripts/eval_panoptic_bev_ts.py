import os
import argparse
import shutil
import time
from collections import OrderedDict
import tensorboardX as tensorboard
import torch
import torch.utils.data as data
from torch import distributed
import inplace_abn
from inplace_abn import ABN
# import torch_tensorrt

from panoptic_bev.config import load_config
from panoptic_bev.custom.custom_grid_sample import custom_grid_sample

from panoptic_bev.data.dataset import BEVKitti360Dataset, BEVTransform, BEVNuScenesDataset
from panoptic_bev.data.misc import iss_collate_fn
from panoptic_bev.data.sampler import DistributedARBatchSampler

from panoptic_bev.modules.ms_transformer import MultiScaleTransformerVF
from panoptic_bev.modules.heads import FPNSemanticHeadDPC as FPNSemanticHeadDPC, FPNMaskHead, RPNHead

from panoptic_bev.models.backbone_edet.efficientdet import EfficientDet
from panoptic_bev.models.panoptic_bev_ts import PanopticBevNetTs, NETWORK_INPUTS
from panoptic_bev.models.panoptic_bev_jit import PanopticBevNetJIT, NETWORK_INPUTS

from panoptic_bev.algos.transformer import TransformerVFAlgo, TransformerVFLoss, TransformerRegionSupervisionLoss
from panoptic_bev.algos.instance_seg_onnx import InstanceSegAlgoFPN_ONNX
from panoptic_bev.algos.rpn_ts import RPNAlgoFPN_JIT
from panoptic_bev.algos.semantic_seg import SemanticSegLoss, SemanticSegAlgo
from panoptic_bev.algos.po_fusion import PanopticLoss, PanopticFusionAlgo

from panoptic_bev.utils.meters import AverageMeter, ConfusionMatrixMeter
from panoptic_bev.utils.misc import config_to_string, norm_act_from_config
from panoptic_bev.utils.parallel import DistributedDataParallel
from panoptic_bev.utils.snapshot import resume_from_snapshot, pre_train_from_snapshots
from panoptic_bev.utils.snapshot import resume_from_snapshot, pre_train_from_snapshots
from panoptic_bev.utils.sequence import pad_packed_images
from panoptic_bev.utils.panoptic import compute_panoptic_test_metrics, panoptic_post_processing, get_panoptic_scores

from panoptic_bev.utils.BevVisualizer import BevVisualizer
from panoptic_bev.utils.ValidMask import ValidMask
from panoptic_bev.utils.Nuscenes_tools import BEVTransformV2, BEVNuScenesDatasetV2
from panoptic_bev.utils.fuse_conv_bn import FuseConvBn
from panoptic_bev.utils import plogging
logger = plogging.get_logger()
# from thop import profile

g_run_multi_view = True
g_run_model_jit = False

parser = argparse.ArgumentParser(description="Panoptic BEV Evaluation Script")
parser.add_argument("--local_rank", required=True, type=int)
parser.add_argument("--run_name", required=True, type=str, help="Name of the run for creating the folders")
parser.add_argument("--project_root_dir", required=True, type=str, help="The root directory of the project")
parser.add_argument("--seam_root_dir", required=True, type=str, help="Seamless dataset directory")
parser.add_argument("--dataset_root_dir", required=True, type=str, help="Kitti360/nuScenes directory")
parser.add_argument("--mode", required=True, type=str, help="'train' the model or 'test' the model")
parser.add_argument("--test_dataset", type=str, choices=['Kitti360', 'nuScenes'], help="Dataset for inference")
parser.add_argument("--resume", metavar="FILE", type=str, help="Resume training from given file", nargs="?")
parser.add_argument("--eval", action="store_true", help="Do a single validation run")
parser.add_argument("--pre_train", type=str, nargs="*",
                    help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                         "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                         "'body', 'rpn_head', 'roi_head' or 'sem_head'. In that case only that part of the network "
                         "will be loaded from the snapshot")
parser.add_argument("--config", required=True, type=str, help="Path to configuration file")
parser.add_argument("--debug", type=bool, default=False, help="Should the program run in 'debug' mode?")
parser.add_argument("--freeze_modules", nargs='+', default=[], help="The modules to freeze. Default is empty")

def log_info(msg, *args, **kwargs):
    if "debug" in kwargs.keys():
        print(msg % args)
    else:
        if distributed.get_rank() == 0:
            plogging.get_logger().info(msg, *args, **kwargs)


def log_miou(label, miou, classes):
    # logger = plogging.get_logger()
    padding = max(len(cls) for cls in classes)

    logger.info("---------------- {} ----------------".format(label))
    for miou_i, class_i in zip(miou, classes):
        logger.info(("{:>" + str(padding) + "} : {:.5f}").format(class_i, miou_i.item()))


def log_scores(label, scores):
    # logger = plogging.get_logger()
    padding = max(len(cls) for cls in scores.keys())

    logger.info("---------------- {} ----------------".format(label))
    for score_label, score_value in scores.items():
        logger.info(("{:>" + str(padding) + "} : {:.5f}").format(score_label, score_value.item()))


def make_config(args, config_file):
    log_info("Loading configuration from %s", config_file)
    conf = load_config(config_file)

    log_info("\n%s", config_to_string(conf))
    return conf


def create_run_directories(args, rank):
    root_dir = args.project_root_dir
    experiment_dir = os.path.join(root_dir, "experiments")
    if args.mode == "train":
        run_dir = os.path.join(experiment_dir, "bev_train_{}".format(args.run_name))
    elif args.mode == "test":
        run_dir = os.path.join(experiment_dir, "bev_test_{}".format(args.run_name))
    else:
        raise RuntimeError("Invalid choice. --mode must be either 'train' or 'test'")
    saved_models_dir = os.path.join(run_dir, "saved_models")
    log_dir = os.path.join(run_dir, "logs")
    vis_dir = os.path.join(run_dir, "vis")
    config_file = os.path.join(run_dir, args.config)

    # Create the directory
    if rank == 0 and (not os.path.exists(experiment_dir)):
        os.mkdir(experiment_dir)
    if rank == 0:
        assert not os.path.exists(run_dir), "Run folder already found! Delete it to reuse the run name."

    if rank == 0:
        os.mkdir(run_dir)
        os.mkdir(saved_models_dir)
        os.mkdir(log_dir)
        os.mkdir(vis_dir)

    # Copy the config file into the folder
    config_path = os.path.join(experiment_dir, "config", args.config)
    if rank == 0:
        shutil.copyfile(config_path, config_file)

    return log_dir, saved_models_dir, config_path, vis_dir

def make_dataloader_v2(args, config, rank, world_size):
    dl_config = config['dataloader']
    log_info("Creating test dataloader for {} dataset, rank: {}, world_size: {}".format(args.test_dataset, rank, world_size), debug=args.debug)
    # Evaluation datalaader
    tfv2 = BEVTransformV2(shortest_size=dl_config.getint("shortest_size"),
                          longest_max_size=dl_config.getint("longest_max_size"),
                          rgb_mean=dl_config.getstruct("rgb_mean"),
                          rgb_std=dl_config.getstruct("rgb_std"),
                          front_resize=dl_config.getstruct("front_resize"))
    datasetv2 = BEVNuScenesDatasetV2(nuscenes_version="v1.0-mini", nuscenes_root_dir="/home/hugoliu/github/dataset/nuscenes/mini", 
                transform=tfv2, bev_size=dl_config.getstruct("bev_crop"))
    test_sampler = DistributedARBatchSampler(datasetv2, dl_config.getint("val_batch_size"), world_size, rank, False)
    test_dl = torch.utils.data.DataLoader(datasetv2,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          drop_last=False,
                                          num_workers=1)
    return test_dl


def make_dataloader(args, config, rank, world_size):
    dl_config = config['dataloader']

    log_info("Creating test dataloader for {} dataset, rank: {}, world_size: {}".format(args.test_dataset, rank, world_size), debug=args.debug)

    # Evaluation datalaader
    test_tf = BEVTransform(shortest_size=dl_config.getint("shortest_size"),
                          longest_max_size=dl_config.getint("longest_max_size"),
                          rgb_mean=dl_config.getstruct("rgb_mean"),
                          rgb_std=dl_config.getstruct("rgb_std"),
                          front_resize=dl_config.getstruct("front_resize"),
                          bev_crop=dl_config.getstruct("bev_crop"))

    if args.test_dataset == "Kitti360":
        test_db = BEVKitti360Dataset(seam_root_dir=args.seam_root_dir, dataset_root_dir=args.dataset_root_dir,
                                    split_name=dl_config['val_set'], transform=test_tf)
    elif args.test_dataset == "nuScenes":
        test_db = BEVNuScenesDataset(seam_root_dir=args.seam_root_dir, dataset_root_dir=args.dataset_root_dir,
                                    split_name=dl_config['val_set'], transform=test_tf)
        # set extrinsics and valid_msk into BEVNuScenesDataset
        cam_config = config['cameras']
        test_db.set_extrinsics(cam_config.getstruct('extrinsics'))
        Z_out=int(dl_config.getstruct("bev_crop")[1] * dl_config.getfloat('scale'))
        W_out=int(dl_config.getstruct('bev_crop')[0] * dl_config.getfloat('scale'))
        valid_msk = ValidMask()
        test_db.set_validmsk(valid_msk.generate(h=W_out, w=Z_out, fov_l=30, fov_r=30))

    if not args.debug:
        test_sampler = DistributedARBatchSampler(test_db, dl_config.getint("val_batch_size"), world_size, rank, False)
        test_dl = torch.utils.data.DataLoader(test_db,
                                             batch_sampler=test_sampler,
                                             collate_fn=iss_collate_fn,
                                             pin_memory=True,
                                             num_workers=dl_config.getint("val_workers"))
    else:
        test_dl = torch.utils.data.DataLoader(test_db,
                                             batch_size=dl_config.getint("val_batch_size"),
                                             collate_fn=iss_collate_fn,
                                             pin_memory=True,
                                             num_workers=dl_config.getint("val_workers"))

    return test_dl


def make_model(args, config, num_thing, num_stuff):
    base_config = config["base"]
    fpn_config = config["fpn"]
    transformer_config = config['transformer']
    rpn_config = config['rpn']
    roi_config = config['roi']
    sem_config = config['sem']
    cam_config = config['cameras']
    dl_config = config['dataloader']

    num_stuff = num_stuff
    num_thing = num_thing
    classes = {"total": num_thing + num_stuff, "stuff": num_stuff, "thing": num_thing}

    # BN + activation
    if not args.debug:
        norm_act_static, norm_act_dynamic = norm_act_from_config(base_config)
    else:
        norm_act_static, norm_act_dynamic = ABN, ABN


    # Create the backbone
    model_compount_coeff = int(base_config["base"][-1])
    model_name = "efficientdet-d{}".format(model_compount_coeff)
    log_info("Creating backbone model %s", base_config["base"], debug=args.debug)
    body = EfficientDet(compound_coef=model_compount_coeff)
    ignore_layers = ['bifpn.0.p5_to_p6', 'bifpn.0.p6_to_p7', 'bifpn.0.p3_down_channel', "bifpn.0.p4_down_channel",
                     "bifpn.0.p5_down_channel", "bifpn.0.p6_down_channel"]
    body = EfficientDet.from_pretrained(body, model_name, ignore_layers=ignore_layers)

    # body_ts = torch.jit.script(body)
    # torch.jit.save(body_ts, "./body_ts.pt")

    # The transformer operates only on a single scale
    extrinsics = cam_config.getstruct('extrinsics')
    extrinsics_t = torch.tensor([extrinsics["translation"], extrinsics["rotation"]], dtype=torch.float)
    # logger.debug("extrinsics_t: {}-{}".format(extrinsics_t[0], extrinsics_t[1]))
    bev_params = cam_config.getstruct('bev_params')
    tfm_scales = transformer_config.getstruct("tfm_scales")

    bev_transformer = MultiScaleTransformerVF(transformer_config.getint("in_channels"),
                                              transformer_config.getint("tfm_channels"),
                                              transformer_config.getint("bev_ms_channels"),
                                              extrinsics_t, bev_params,
                                              H_in=dl_config.getstruct("front_resize")[0] * dl_config.getfloat("scale"),
                                              W_in=dl_config.getstruct('front_resize')[1] * dl_config.getfloat('scale'),
                                              Z_out=dl_config.getstruct("bev_crop")[1] * dl_config.getfloat('scale'),
                                              W_out=dl_config.getstruct('bev_crop')[0] * dl_config.getfloat('scale'),
                                              tfm_scales=tfm_scales,
                                              use_init_theta=transformer_config['use_init_theta'],
                                              norm_act=norm_act_static)

    vf_loss = None
    region_supervision_loss = None
    transformer_algo = None

    W_out = int(dl_config.getstruct("bev_crop")[0] * dl_config.getfloat("scale"))
    Z_out = int(dl_config.getstruct("bev_crop")[1] * dl_config.getfloat("scale"))

    # Create RPN
    # proposal_generator = ProposalGenerator(rpn_config.getfloat("nms_threshold"),
    #                                        rpn_config.getint("num_pre_nms_train"),
    #                                        rpn_config.getint("num_post_nms_train"),
    #                                        rpn_config.getint("num_pre_nms_val"),
    #                                        rpn_config.getint("num_post_nms_val"),
    #                                        rpn_config.getint("min_size"))
    anchor_matcher = None
    rpn_loss = None
    anchor_scales = [int(scale) for scale in rpn_config.getstruct('anchor_scale')]
    anchor_ratios = [float(ratio) for ratio in rpn_config.getstruct('anchor_ratios')]
    # rpn_algo = RPNAlgoFPN(proposal_generator, anchor_matcher, rpn_loss, anchor_scales, anchor_ratios,
    #                       fpn_config.getstruct("out_strides"), rpn_config.getint("fpn_min_level"),
    #                       rpn_config.getint("fpn_levels"))
    rpn_head = RPNHead(transformer_config.getint("bev_ms_channels"), int(len(anchor_scales) * len(anchor_ratios)), 1,
                       rpn_config.getint("hidden_channels"), norm_act_dynamic)

    rpn_valid_size = [torch.tensor([W_out, Z_out*2])]
    rpn_algo = RPNAlgoFPN_JIT(rpn_head, rpn_config.getfloat("nms_threshold"), rpn_config.getint("num_pre_nms_val"),
                                rpn_config.getint("num_post_nms_val"), rpn_config.getint("min_size"),
                                anchor_scales, anchor_ratios,
                                fpn_config.getstruct("out_strides"), rpn_config.getint("fpn_min_level"),
                                rpn_config.getint("fpn_levels"), rpn_valid_size)

    # Create instance segmentation network
    # bbx_prediction_generator = BbxPredictionGenerator(roi_config.getfloat("nms_threshold"),
    #                                                   roi_config.getfloat("score_threshold"),
    #                                                   roi_config.getint("max_predictions"),
    #                                                   dataset_name=args.test_dataset)
    msk_prediction_generator = None #MskPredictionGenerator()
    roi_size = roi_config.getstruct("roi_size")
    proposal_matcher = None
            # ProposalMatcher(classes,
            # roi_config.getint("num_samples"),
            # roi_config.getfloat("pos_ratio"),
            # roi_config.getfloat("pos_threshold"),
            # roi_config.getfloat("neg_threshold_hi"),
            # roi_config.getfloat("neg_threshold_lo"),
            # roi_config.getfloat("void_threshold"))
    bbx_loss = None
    msk_loss = None
    lbl_roi_size = tuple(s * 2 for s in roi_size)
    # roi_algo = InstanceSegAlgoFPN(bbx_prediction_generator, msk_prediction_generator, proposal_matcher, bbx_loss, msk_loss, classes,
    #                               roi_config.getstruct("bbx_reg_weights"), roi_config.getint("fpn_canonical_scale"),
    #                               roi_config.getint("fpn_canonical_level"), roi_size, roi_config.getint("fpn_min_level"),
    #                               roi_config.getint("fpn_levels"), lbl_roi_size, roi_config.getboolean("void_is_background"), args.debug)

    roi_head = FPNMaskHead(transformer_config.getint("bev_ms_channels"), classes, roi_size, norm_act=norm_act_dynamic)
    roi_algo = InstanceSegAlgoFPN_ONNX(bbx_loss, msk_loss, roi_config.getstruct("bbx_reg_weights"), 
                                    roi_config.getint("fpn_canonical_scale"), roi_config.getint("fpn_canonical_level"), roi_size, 
                                    roi_config.getint("fpn_min_level"), roi_config.getint("fpn_levels"),
                                    roi_config.getfloat("nms_threshold"), roi_config.getfloat("score_threshold"), roi_config.getint("max_predictions"), rpn_valid_size)

    # roi_algo = torch.jit.script(roi_algo)
    # torch.jit.save(roi_algo, "../jit/roi_algo.pt")

    # Create semantic segmentation network
    out_shape = (W_out, Z_out)
    sem_loss = None
    sem_img_size = torch.tensor([W_out, Z_out*2])
    sem_head = FPNSemanticHeadDPC(transformer_config.getint("bev_ms_channels"),
                                  sem_config.getint("fpn_min_level"),
                                  sem_config.getint("fpn_levels"),
                                  classes["total"],
                                  out_size=out_shape,
                                  pooling_size=sem_config.getstruct("pooling_size"),
                                  norm_act=norm_act_static)
    sem_algo = SemanticSegAlgo(sem_head, sem_loss, classes["total"], sem_img_size)

    # Panoptic fusion algorithm
    po_loss = None
    po_fusion_algo = PanopticFusionAlgo(po_loss, classes["stuff"], classes["thing"], 1)

    # po_fusion_ts = PO_FUSION_TS()
    # po_fusion_jit = torch.jit.script(po_fusion_ts)
    # torch.jit.save(po_fusion_jit, "../jit/po_fusion_2.pt")

    # panoptic_bev_jit = PanopticBevNetJIT(out_shape=out_shape, tfm_scales=tfm_scales)
    # model_jit = torch.jit.script(panoptic_bev_jit)
    # frozen_model = torch.jit.freeze(model_jit)
    # torch.jit.save(frozen_model, "../jit/panoptic_bev_gpu_2.pt")

    # Create the BEV network
    # return panoptic_bev_jit
    return PanopticBevNetTs(body, bev_transformer, rpn_head, roi_head, sem_head, transformer_algo, rpn_algo, roi_algo,
                          sem_algo, po_fusion_algo, args.test_dataset, classes=classes,
                          front_vertical_classes=transformer_config.getstruct("front_vertical_classes"),
                          front_flat_classes=transformer_config.getstruct("front_flat_classes"),
                          bev_vertical_classes=transformer_config.getstruct('bev_vertical_classes'),
                          bev_flat_classes=transformer_config.getstruct("bev_flat_classes"),
                          out_shape=out_shape,
                          tfm_scales=tfm_scales)


def freeze_modules(args, model):
    for module in args.freeze_modules:
        print("Freezing module: {}".format(module))
        for name, param in model.named_parameters():
            if name.startswith(module):
                param.requires_grad = False

    # Freeze the dummy parameters
    for name, param in model.named_parameters():
        if name.endswith("dummy.weight"):
            param = torch.ones_like(param)
            param.requires_grad = False
            print("Freezing layer: {}".format(name))
        elif name.endswith("dummy.bias"):
            param = torch.zeros_like(param)
            param.requires_grad = False
            print("Freezing layer: {}".format(name))

    return model


def log_iter(mode, meters, time_meters, results, metrics, batch=True, **kwargs):
    assert mode in ['train', 'val', 'test'], "Mode must be either 'train', 'val', or 'test'!"
    iou = ["sem_conf"]

    log_entries = []

    if kwargs['lr'] is not None:
        log_entries = [("lr", kwargs['lr'])]

    meters_keys = list(meters.keys())
    meters_keys.sort()
    for meter_key in meters_keys:
        if meter_key in iou:
            log_key = meter_key
            log_value = meters[meter_key].iou.mean().item()
        else:
            if not batch:
                log_value = meters[meter_key].mean.item()
            else:
                log_value = meters[meter_key]
            log_key = meter_key

        log_entries.append((log_key, log_value))

    time_meters_keys = list(time_meters.keys())
    time_meters_keys.sort()
    for meter_key in time_meters_keys:
        log_key = meter_key
        if not batch:
            log_value = time_meters[meter_key].mean.item()
        else:
            log_value = time_meters[meter_key]
        log_entries.append((log_key, log_value))

    if metrics is not None:
        metrics_keys = list(metrics.keys())
        metrics_keys.sort()
        for metric_key in metrics_keys:
            log_key = metric_key
            if not batch:
                log_value = metrics[log_key].mean.item()
            else:
                log_value = metrics[log_key]
            log_entries.append((log_key, log_value))

    plogging.iteration(kwargs["summary"], mode, kwargs["global_step"], kwargs["epoch"] + 1, kwargs["num_epochs"],
                      kwargs['curr_iter'], kwargs['num_iters'], OrderedDict(log_entries))

def make_result_dict(results_list):
    results_dict = OrderedDict()
    results_dict['bbx_pred'] = results_list[0]
    results_dict['cls_pred'] = results_list[1]
    results_dict['obj_pred'] = results_list[2]
    results_dict["sem_pred"] = results_list[3]
    logger.info("panoptic_bev output, bbx_pred: {}, cls_pred: {}, obj_pred: {}, sem_pred: {}".format(results_dict['bbx_pred'].shape, results_dict['cls_pred'].shape, results_dict['obj_pred'].shape, results_dict["sem_pred"].shape))
    results_dict['po_pred'] = results_list[4]
    results_dict['po_class'] = results_list[5]
    results_dict['po_iscrowd'] = results_list[6]
    logger.info("panoptic_bev output, po_pred: {}, po_class: {}, po_iscrowd: {}".format(results_dict['po_pred'].shape, results_dict['po_class'], results_dict['po_iscrowd']))
    return results_dict

def test_jit_model():
    jit_path = "../jit/panoptic_bev_gpu_768.pt"
    device=torch.device('cuda:0')
    # device=torch.device('cpu')
    B = 1
    N = 6
    img = torch.rand(size=(B,N,3,448,768), dtype=torch.float, device=device)
    calib = torch.rand(size=(B,N,3,3), dtype=torch.float, device=device)
    extrinsics = torch.rand(size=(B,N,2,3), dtype=torch.float, device=device)
    valid_msk = torch.randint(low=0, high=2, size=(B,N,896,768), dtype=torch.uint8, device=device)

    print("[{}], load jit model: {}".format(time.time(), jit_path))
    jit_model = torch.jit.load(jit_path)
    jit_model.eval()

    # onnx_path = "../jit/panoptic_bev_gpu_768.onnx"
    # torch.onnx.export(jit_model, (img, calib, extrinsics, valid_msk), onnx_path, opset_version=12)

    # trt_model_fp32 = torch_tensorrt.compile(
    #     module=jit_model, 
    #     inputs=[torch_tensorrt.Input((B,N,3,448,768), dtype=torch.float32), torch_tensorrt.Input((B,N,3,3), dtype=torch.float32), torch_tensorrt.Input((B,N,2,3), dtype=torch.float32), torch_tensorrt.Input((B,N,896,768), dtype=torch.float32)], 
    #     enabled_precisions={torch.float32},
    #     workspace_size=1<<22
    # )
    # torch.jit.save(trt_model_fp32, "./trt_model_fp32.trt")
    # print("trt_model_fp32 saved")

    # jit_model_cpu = jit_model_gpu.cpu()
    # warm-up stage: do optimization for given input
    with torch.jit.optimized_execution(True):
        with torch.no_grad():
            for idx in range(2):
                results = jit_model(img, calib, extrinsics, valid_msk)

    LOOP = 100
    start_time = time.time()
    with torch.jit.optimized_execution(True):
        with torch.no_grad():
            for idx in range(LOOP):
                print("[{}], idx-{}, input img: {}".format(time.time(), idx, img.shape))
                results = jit_model(img, calib, extrinsics, valid_msk)
                # print("test jit model, results: {}".format(results))
                print("[{}], idx-{}, inference done".format(time.time(), idx))
    end_time = time.time()
    print("{}-loop total inference time: {}, average time: {}".format(LOOP, (end_time-start_time), (end_time-start_time)/LOOP))

def test(model, dataloader, **varargs):
    global g_bev_visualizer

    if g_run_model_jit:
        model_jit = torch.jit.load("../jit/panoptic_bev_gpu_list_768.pt")
        model_jit.eval()
    else:
        fuser = FuseConvBn()
        fuser.do_bn_fusion_v2(model, bn_name="SyncBatchNorm")
        fuser.do_bn_fusion_v2(model, bn_name="InPlaceABNSync")
        # logger.info("final model after fusion: {}".format(model))
        model.eval()

    num_stuff = dataloader.dataset.num_stuff
    num_thing = dataloader.dataset.num_thing
    num_classes = num_stuff + num_thing

    test_meters = {
        "loss": AverageMeter(()),
        "obj_loss": AverageMeter(()),
        "bbx_loss": AverageMeter(()),
        "roi_cls_loss": AverageMeter(()),
        "roi_bbx_loss": AverageMeter(()),
        "roi_msk_loss": AverageMeter(()),
        "sem_loss": AverageMeter(()),
        "po_loss": AverageMeter(()),
        "sem_conf": ConfusionMatrixMeter(num_classes),
        "vf_loss": AverageMeter(()),
        "v_region_loss": AverageMeter(()),
        "f_region_loss": AverageMeter(())
    }

    time_meters = {"data_time": AverageMeter(()),
                   "batch_time": AverageMeter(())}

    # Inference metrics
    test_metrics = {"po_miou": AverageMeter(()), "sem_miou": AverageMeter(()),
                   "pq": AverageMeter(()), "pq_stuff": AverageMeter(()), "pq_thing": AverageMeter(()),
                   "sq": AverageMeter(()), "sq_stuff": AverageMeter(()), "sq_thing": AverageMeter(()),
                   "rq": AverageMeter(()), "rq_stuff": AverageMeter(()), "rq_thing": AverageMeter(())}

    # Accumulators for AP, mIoU and panoptic computation
    panoptic_buffer = torch.zeros(4, num_classes, dtype=torch.double)
    po_conf_mat = torch.zeros(256, 256, dtype=torch.double)
    sem_conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.double)

    logger.debug("num_stuff: {}, num_thing: {}, panoptic_buffer: {}, po_conf_mat: {}, sem_conf_mat: {}".format(num_stuff, num_thing, panoptic_buffer.shape, po_conf_mat.shape, sem_conf_mat.shape))

    data_time = time.time()

    for it, sample in enumerate(dataloader):
        # eval with front-100 images
        if it > 10:
            break

        do_loss=False
        with torch.no_grad():
            if not g_run_multi_view:
                sample = {k: sample[k].cuda(device=varargs['device'], non_blocking=True) for k in NETWORK_INPUTS}
            else:
                sample_cuda = {}
                for k in NETWORK_INPUTS:
                    val = sample[k]
                    sample_cuda[k] = val.cuda(device=varargs['device'], non_blocking=True)
                sample = sample_cuda

            # sample['calib'], _ = pad_packed_images(sample['calib'])

            time_meters['data_time'].update(torch.tensor(time.time() - data_time))
            batch_time = time.time()

            # Run network
            loop = 1
            if it > 4:
                loop = 100
            # results = model(**sample)
            logger.debug("inputs img: {}, calib: {}, extrinsics: {}, valid_msk: {}".format(sample["img"].shape, sample["calib"].shape, sample["extrinsics"].shape, sample["valid_msk"].shape))
            inputs = (sample["img"], sample["calib"], sample["extrinsics"], sample["valid_msk"])
            # imgs = torch.rand(size=(1,6,3,1024,1920), dtype=torch.float, device=sample["img"].device)
            if g_run_model_jit:
                results = model_jit(sample["img"], sample["calib"], sample["extrinsics"], sample["valid_msk"])
            else:
                results = model(sample["img"], sample["calib"], sample["extrinsics"], sample["valid_msk"])
            # break

            model_ts = torch.jit.trace(model, inputs, check_trace=True, strict=True)
            # frozen_model = torch.jit.optimize_for_inference(model_ts)
            # torch.jit.save(frozen_model, "../jit/panoptic_bev_gpu_list_768.pt")
            # logger.info("torchscript model saved to ../jit/panoptic_bev_gpu_list_768.pt")
            # return

            torch.onnx.export(
                model=model_ts, 
                args=(sample["img"], sample["calib"], sample["extrinsics"], sample["valid_msk"]), 
                f="../onnx/multi_view_perception_768.onnx", 
                custom_opsets={"custom_domain": 1},
                opset_version=13, verbose=True, do_constant_folding=False)
            return

            ### save panoptic_bev_cpu.pt due to Perspective2OrthographicWarper.forward()
            # device=torch.device('cpu')
            # model_cpu = model.cpu()
            # inputs_cpu = (inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device))
            # model_cpu_ts = torch.jit.trace(model_cpu, inputs_cpu, strict=False, check_trace=True)
            # torch.jit.save(model_cpu_ts, "../jit/panoptic_bev_cpu.pt")
            # break

            # macs, params = profile(model, inputs)
            # logger.info("thop profile, macs: {}, params: {}".format(macs, params))

            if not varargs['debug']:
                distributed.barrier()

            results = make_result_dict(results)
            g_bev_visualizer.plot_bev(sample, it, results, show_po=True)
            # break
            # for idx, (path, submodule) in enumerate(model.named_modules()):
            #     logger.info("named_modules-{} - {} - {}: {}".format(idx, path, submodule.__class__.__name__, submodule))
            # break            

    # Finalise Panoptic mIoU computation
    po_conf_mat = po_conf_mat.to(device=varargs["device"])
    if not varargs['debug']:
        distributed.all_reduce(po_conf_mat, distributed.ReduceOp.SUM)
    po_conf_mat = po_conf_mat.cpu()[:num_classes, :]
    po_intersection = po_conf_mat.diag()
    po_union = ((po_conf_mat.sum(dim=1) + po_conf_mat.sum(dim=0)[:num_classes] - po_conf_mat.diag()) + 1e-8)
    po_miou = po_intersection / po_union

    # Finalise semantic mIoU computation
    sem_conf_mat = sem_conf_mat.to(device=varargs['device'])
    if not varargs['debug']:
        distributed.all_reduce(sem_conf_mat, distributed.ReduceOp.SUM)
    sem_conf_mat = sem_conf_mat.cpu()[:num_classes, :]
    sem_intersection = sem_conf_mat.diag()
    sem_union = ((sem_conf_mat.sum(dim=1) + sem_conf_mat.sum(dim=0)[:num_classes] - sem_conf_mat.diag()) + 1e-8)
    sem_miou = sem_intersection / sem_union

    # Save the metrics
    scores = {}
    scores['po_miou'] = po_miou.mean()
    scores['sem_miou'] = sem_miou.mean()
    scores = get_panoptic_scores(panoptic_buffer, scores, varargs["device"], num_stuff, varargs['debug'])
    # Update the inference metrics meters
    for key in test_metrics.keys():
        if key in scores.keys():
            if scores[key] is not None:
                test_metrics[key].update(scores[key].cpu())

    # Log results
    log_info("Evaluation done", debug=varargs['debug'])
    if varargs["summary"] is not None and do_loss:
        log_iter("val", test_meters, time_meters, None, test_metrics, batch=False, summary=varargs['summary'],
                 global_step=varargs['global_step'], curr_iter=len(dataloader), num_iters=len(dataloader),
                 epoch=varargs['epoch'], num_epochs=varargs['num_epochs'], lr=None)

    log_miou("Semantic mIoU", sem_miou, dataloader.dataset.categories)
    log_scores("Panoptic Scores", scores)

    return scores['pq'].item()

def main_jit(args):
    global g_bev_visualizer
    if not args.debug:
        # Initialize multi-processing
        distributed.init_process_group(backend='nccl', init_method='env://')
        device_id, device = args.local_rank, torch.device(args.local_rank)
        rank, world_size = distributed.get_rank(), distributed.get_world_size()
        torch.cuda.set_device(device_id)
    else:
        rank = 0
        world_size = 1
        device_id, device = rank, torch.device(rank+3)

    # Create directories
    if not args.debug:
        log_dir, saved_models_dir, config_file, vis_dir = create_run_directories(args, rank)
    else:
        config_file = os.path.join(args.project_root_dir, "experiments", "config", args.config)

    # Load configuration
    config = make_config(args, config_file)

    g_bev_visualizer = BevVisualizer(config, vis_dir, n_imgs=6 if g_run_multi_view else 1)

    # Initialize logging only for rank 0
    if not args.debug and rank == 0:
        plogging.init(log_dir, "train" if args.mode == 'train' else "test")
        summary = tensorboard.SummaryWriter(log_dir)
    else:
        summary = None

    # Create dataloaders
    if g_run_multi_view:
        test_dataloader = make_dataloader_v2(args, config, rank, world_size)
    else:
        test_dataloader = make_dataloader(args, config, rank, world_size)

    # Create model
    model = make_model(args, config, test_dataloader.dataset.num_thing, test_dataloader.dataset.num_stuff)
    # model.load_trained_params()

    # Init GPU stuff
    if not args.debug:
        torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
        model = model.cuda(device)
        # model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id, find_unused_parameters=True)
    else:
        model = model.cuda(device)

    if args.resume:
        epoch = 0
        global_step = 0

        log_info("Evaluating epoch %d", epoch + 1, debug=args.debug)
        score = test(model, test_dataloader, device=device, summary=summary, global_step=global_step,
                     epoch=epoch, num_epochs=epoch+1, log_interval=config["general"].getint("log_interval"),
                     loss_weights=config['optimizer'].getstruct("loss_weights"),
                     front_vertical_classes=config['transformer'].getstruct('front_vertical_classes'),
                     front_flat_classes=config['transformer'].getstruct('front_flat_classes'),
                     bev_vertical_classes=config['transformer'].getstruct('bev_vertical_classes'),
                     bev_flat_classes=config['transformer'].getstruct('bev_flat_classes'),
                     rgb_mean=config['dataloader'].getstruct('rgb_mean'),
                     rgb_std=config['dataloader'].getstruct('rgb_std'),
                     img_scale=config['dataloader'].getfloat('scale'),
                     debug=args.debug)

def main(args):
    global g_bev_visualizer
    if not args.debug:
        # Initialize multi-processing
        distributed.init_process_group(backend='nccl', init_method='env://')
        device_id, device = args.local_rank, torch.device(args.local_rank)
        rank, world_size = distributed.get_rank(), distributed.get_world_size()
        torch.cuda.set_device(device_id)
    else:
        rank = 0
        world_size = 1
        device_id, device = rank, torch.device(rank+3)

    # Create directories
    if not args.debug:
        log_dir, saved_models_dir, config_file, vis_dir = create_run_directories(args, rank)
    else:
        config_file = os.path.join(args.project_root_dir, "experiments", "config", args.config)

    # Load configuration
    config = make_config(args, config_file)

    g_bev_visualizer = BevVisualizer(config, vis_dir, n_imgs=6 if g_run_multi_view else 1)

    # Initialize logging only for rank 0
    if not args.debug and rank == 0:
        plogging.init(log_dir, "train" if args.mode == 'train' else "test")
        summary = tensorboard.SummaryWriter(log_dir)
    else:
        summary = None

    # Create dataloaders
    if g_run_multi_view:
        test_dataloader = make_dataloader_v2(args, config, rank, world_size)
    else:
        test_dataloader = make_dataloader(args, config, rank, world_size)

    # Create model
    model = make_model(args, config, test_dataloader.dataset.num_thing, test_dataloader.dataset.num_stuff)
    # model.load_trained_params()

    # Freeze modules based on the argument inputs
    model = freeze_modules(args, model)

    if args.resume:
        assert not args.pre_train, "resume and pre_train are mutually exclusive"
        log_info("Loading snapshot from %s", args.resume, debug=args.debug)
        snapshot = resume_from_snapshot(model, args.resume, ["body", "transformer", "rpn_head", "roi_head", "sem_head"])
        # snapshot = resume_from_snapshot(model, args.resume, ["transformer", "rpn_head", "roi_head", "sem_head"])
    elif args.pre_train:
        assert not args.resume, "resume and pre_train are mutually exclusive"
        log_info("Loading pre-trained model from %s", args.pre_train, debug=args.debug)
        pre_train_from_snapshots(model, args.pre_train, ["body", "transformer", "rpn_head", "roi_head", "sem_head"], rank)
    else:
        raise Exception("Either --resume or --pre_train need to be defined")
        snapshot = None

    # Init GPU stuff
    if not args.debug:
        torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Convert batch norm to SyncBatchNorm
        model = model.cuda(device)
        # model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id, find_unused_parameters=True)
    else:
        model = model.cuda(device)

    if args.resume:
        epoch = snapshot["training_meta"]["epoch"] + 1
        global_step = snapshot["training_meta"]["global_step"]
        del snapshot

        log_info("Evaluating epoch %d", epoch + 1, debug=args.debug)
        score = test(model, test_dataloader, device=device, summary=summary, global_step=global_step,
                     epoch=epoch, num_epochs=epoch+1, log_interval=config["general"].getint("log_interval"),
                     loss_weights=config['optimizer'].getstruct("loss_weights"),
                     front_vertical_classes=config['transformer'].getstruct('front_vertical_classes'),
                     front_flat_classes=config['transformer'].getstruct('front_flat_classes'),
                     bev_vertical_classes=config['transformer'].getstruct('bev_vertical_classes'),
                     bev_flat_classes=config['transformer'].getstruct('bev_flat_classes'),
                     rgb_mean=config['dataloader'].getstruct('rgb_mean'),
                     rgb_std=config['dataloader'].getstruct('rgb_std'),
                     img_scale=config['dataloader'].getfloat('scale'),
                     debug=args.debug)

if __name__ == "__main__":
    main(parser.parse_args())
    # test_jit_model()
