mode=$1

if [ $mode = 'train' ]
then
    python main.py \
    --batch_size 10 \
    --epochs 50 \
    --coco_path ~/Desktop/hdd \
    --dataset_file AssemblyHands \
    --resume old_weights/paper_pose.pth \
    --use_h2o_pth
else
    python main.py \
    --batch_size 10 \
    --coco_path ~/Desktop/hdd \
    --dataset_file AssemblyHands \
    --resume weights/AssemblyHands/49.pth \
    --eval
fi