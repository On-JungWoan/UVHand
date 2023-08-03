mode=$1

if [ $mode = 'train' ]
then
    python main.py \
    --batch_size 10 \
    --epochs 50 \
    --coco_path ~/Desktop/hdd \
    --dataset_file AssemblyHands \
    --resume old_weights/paper_pose.pth \
    --use_h2o_pth \
    --wandb
elif [ $mode = 'viewpoint' ]
then
    for viewpoint in '84346135' '84347414' '84355350' '84358933'
    do
        python main.py \
        --batch_size 1 \
        --coco_path ~/Desktop/hdd \
        --dataset_file AssemblyHands \
        --resume weights/AssemblyHands/49.pth \
        --output_dir ./results/H2O+AHands_50_epoch \
        --test_viewpoint 'ego_images_rectified/val/nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345/HMC_'$viewpoint'_mono10bit' \
        --eval
    done

    for viewpoint in '21110305' '21176623' '21176875' '21179183'
    do
        python main.py \
        --batch_size 1 \
        --coco_path ~/Desktop/hdd \
        --dataset_file AssemblyHands \
        --resume weights/AssemblyHands/49.pth \
        --output_dir ./results/H2O+AHands_50_epoch \
        --test_viewpoint 'ego_images_rectified/val/nusar-2021_action_both_9081-c11b_9081_user_id_2021-02-12_161433/HMC_'$viewpoint'_mono10bit' \
        --eval
    done
else
    python main.py \
    --batch_size 15 \
    --coco_path ~/Desktop/hdd \
    --dataset_file AssemblyHands \
    --resume weights/AssemblyHands/49.pth \
    --output_dir ./results/H2O+AHands_50_epoch \
    --eval
fi