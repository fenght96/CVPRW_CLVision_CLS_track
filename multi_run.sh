CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --use_env multi_detection_faster.py

CUDA_VISIBLE_DEVICES=2 python multi_detection_faster_eval.py