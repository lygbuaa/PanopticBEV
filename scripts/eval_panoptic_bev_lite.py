import os
import argparse
import shutil
import time
from collections import OrderedDict
import torch
import torch.utils.data as data
from torch import distributed
from panoptic_bev.config import load_config
from export.panoptic_bev_export import PanopticBevModel
from panoptic_bev.utils.misc import config_to_string
from panoptic_bev.utils.BevVisualizer import BevVisualizer
from panoptic_bev.utils.Nuscenes_tools import BEVTransformV2, BEVNuScenesDatasetV2
from panoptic_bev.utils import plogging
logger = plogging.get_logger()

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

def make_config(args, config_file):
    logger.info("Loading configuration from %s", config_file)
    conf = load_config(config_file)

    logger.info("\n%s", config_to_string(conf))
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
    logger.info("Creating test dataloader for {} dataset, rank: {}, world_size: {}".format(args.test_dataset, rank, world_size))
    # Evaluation datalaader
    tfv2 = BEVTransformV2(shortest_size=dl_config.getint("shortest_size"),
                          longest_max_size=dl_config.getint("longest_max_size"),
                          rgb_mean=dl_config.getstruct("rgb_mean"),
                          rgb_std=dl_config.getstruct("rgb_std"),
                          front_resize=dl_config.getstruct("front_resize"))
    datasetv2 = BEVNuScenesDatasetV2(nuscenes_version="v1.0-mini", nuscenes_root_dir="/home/hugoliu/github/dataset/nuscenes/mini", 
                transform=tfv2, bev_size=dl_config.getstruct("bev_crop"))
    test_dl = torch.utils.data.DataLoader(datasetv2,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          drop_last=False,
                                          num_workers=1)
    return test_dl, datasetv2

def make_model(args, config):
    dl_config = config['dataloader']
    W_out = int(dl_config.getstruct("bev_crop")[0] * dl_config.getfloat("scale"))
    Z_out = int(dl_config.getstruct("bev_crop")[1] * dl_config.getfloat("scale"))
    out_shape = (W_out, Z_out)
    transformer_config = config['transformer']
    tfm_scales = transformer_config.getstruct("tfm_scales")
    return PanopticBevModel(out_shape, tfm_scales)

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

def test(model, dataloader, **varargs):
    global g_bev_visualizer
    model.eval()
    avg_infer_time = 0.0
    n_infer = 10

    for it, sample in enumerate(dataloader):
        # eval with front-100 images
        if it > n_infer:
            break

        with torch.no_grad():
            sample_cuda = {}
            for k in ["img", "calib", "extrinsics", "valid_msk"]:
                val = sample[k]
                sample_cuda[k] = val.cuda(device=varargs['device'], non_blocking=True)
            sample = sample_cuda
            start_time = time.time()
            results = model(sample["img"])
            avg_infer_time += (time.time() - start_time)

            results = make_result_dict(results)
            g_bev_visualizer.plot_bev(sample, it, results, show_po=True)
    avg_infer_time /= n_infer
    logger.info("total iter: {}, avg_infer_time: {}".format(n_infer, avg_infer_time))

def main(args):
    global g_bev_visualizer
    # Initialize multi-processing
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)
    # Create directories
    log_dir, _, config_file, vis_dir = create_run_directories(args, rank)

    # Load configuration
    config = make_config(args, config_file)

    g_bev_visualizer = BevVisualizer(config, vis_dir, n_imgs=6)

    # Initialize logging only for rank 0
    if not args.debug and rank == 0:
        plogging.init(log_dir, "test")

    # Create dataloaders
    test_dataloader, _ = make_dataloader_v2(args, config, rank, world_size)

    # Create model
    model = make_model(args, config)#, test_dataloader.dataset.num_thing, test_dataloader.dataset.num_stuff, init_generator)
    test(model, test_dataloader, device=device)

if __name__ == "__main__":
    main(parser.parse_args())
