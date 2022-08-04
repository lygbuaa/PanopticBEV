reset
rm -rf /home/hugoliu/github/PanopticBEV/experiments/bev_test_export_eval/
export LD_PRELOAD=/home/hugoliu/github/onnxparser-trt-plugin-sample/TensorRT/build/out/libnvinfer_plugin.so

CUDA_VISIBLE_DEVICES=0 \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_addr="10.11.10.129" --master_port=30000 eval_panoptic_bev_lite.py \
                                    --run_name='export_eval' \
                                    --project_root_dir="/home/hugoliu/github/PanopticBEV" \
                                    --seam_root_dir="/home/hugoliu/github/dataset/panopticbev/nuScenes_panopticbev" \
                                    --dataset_root_dir="/home/hugoliu/github/dataset/nuscenes/trainval" \
                                    --mode=test \
                                    --test_dataset=nuScenes \
                                    --config=nuscenes.ini
unset LD_PRELOAD