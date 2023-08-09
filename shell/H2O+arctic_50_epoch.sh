PUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./tools/run_h2otr.sh \
--num_workers 24 \
--batch_size 16 \
--epochs 50 \
--coco_path .. \
--dataset_file arctic \
--resume paper_pose.pth \
--use_h2o_pth