PUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./tools/run_h2otr.sh \
--num_workers 24 \
--batch_size 128 \
--epochs 50 \
--coco_path .. \
--dataset_file arctic \
--resume old_weights/paper_pose.pth \
--use_h2o_pth \
--wandb



python tools/launch.py \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 12355 --nproc_per_node 4 \
python -u main.py \
--num_workers 24 \
--batch_size 12 \
--epochs 50 \
--coco_path /home/user \
--dataset_file arctic \
--resume old_weights/paper_pose.pth \
--use_h2o_pth \
--wandb