python main.py \
--batch_size 14 \
--epochs 50 \
--coco_path /home/unist/Desktop/hdd \
--dataset_file arctic \
--resume weights/arctic/1.pth \
--start_epoch 2 \
--wandb

python main.py \
--batch_size 1 \
--coco_path /home/unist/Desktop/hdd \
--dataset_file arctic \
--resume weights/arctic/1.pth