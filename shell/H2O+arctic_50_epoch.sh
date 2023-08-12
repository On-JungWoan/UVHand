python main.py \
--batch_size 128 \
--epochs 50 \
--coco_path .. \
--dataset_file arctic \
--resume weights/arctic/0.pth \
--start_epoch 2 \
--wandb

python main.py \
--batch_size 1 \
--coco_path /home/unist/Desktop/hdd \
--dataset_file arctic \
--resume weights/arctic/1.pth

CUDA_VISIBLE_DEVICES=1 \
python main.py \
--coco_path .. \
--dataset_file arctic \
--resume old_weights/paper_pose.pth \
--batch_size 128 \
--num_workers 24 \
--use_h2o_pth \
--debug