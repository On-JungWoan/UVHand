from detectron2.data.datasets import register_coco_instances

# train
register_coco_instances(
    "Assembly_hands_train",{},
    "/home/unist/Desktop/hdd/AssemblyHands/annotations/train.json", "/home/unist/Desktop/hdd/AssemblyHands/ego_images_rectified/train"
)
# val
register_coco_instances(
    "Assembly_hands_val",{},
    "/home/unist/Desktop/hdd/AssemblyHands/annotations/val.json", "/home/unist/Desktop/hdd/AssemblyHands/ego_images_rectified/val"
)