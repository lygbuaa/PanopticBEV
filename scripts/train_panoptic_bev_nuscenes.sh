CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_addr="10.11.10.144" --master_port="30000" train_panoptic_bev.py \
                                    --run_name='train_panoptic_bev' \
                                    --project_root_dir="/home/hugoliu/github/PanopticBEV" \
                                    --seam_root_dir="/home/hugoliu/github/dataset/panopticbev/nuScenes_panopticbev" \
                                    --dataset_root_dir="/home/hugoliu/github/dataset/nuscenes/trainval" \
                                    --mode=train \
                                    --train_dataset=nuScenes \
                                    --val_dataset=nuScenes \
                                    --config=nuscenes.ini
                                    # --resume="/home/hugoliu/github/PanopticBEV/weights/d3_0128_448_768/saved_models/model_latest.pth"
