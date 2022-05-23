reset
rm -rf /home/hugoliu/github/PanopticBEV/experiments/bev_test_generate_torchscript/

CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_addr="10.11.10.129" --master_port=30000 eval_panoptic_bev_ts.py \
                                    --run_name='generate_torchscript' \
                                    --project_root_dir="/home/hugoliu/github/PanopticBEV" \
                                    --seam_root_dir="/home/hugoliu/github/dataset/panopticbev/nuScenes_panopticbev" \
                                    --dataset_root_dir="/home/hugoliu/github/dataset/nuscenes/trainval" \
                                    --mode=test \
                                    --test_dataset=nuScenes \
                                    --resume="/home/hugoliu/github/PanopticBEV/weights/d3_0328_448_768/saved_models/model_best.pth" \
                                    --config=nuscenes.ini \
