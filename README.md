# HRCPose Demo

## FLOPs / Params

python get_flops.py configs/animal_2d_keypoint/HRCPose/ap10k/HRCPose_ap10k-256x256.py --input-shape 256 256 --device cuda:0

## AP-10K Demo

python infer_demo.py --config configs/animal_2d_keypoint/HRCPose/ap10k/HRCPose_ap10k-256x256.py --checkpoint weights/ap10k-pretrain.pth --image-dir data/ap10k/data --coco-json data/ap10k/annotations/ap10k-test-split1.json --out-dir img/ap10k_output --force-legacy

## CM9K (Monkey) Demo

python infer_demo.py --config configs/monkey/HRCPose/HRCPose/HRCPose_monkey-384x288.py --checkpoint weights/monkey.pth --image-dir data/monkey/data --coco-json data/monkey/annotations/val_demo.json --out-dir img/monkey_output --force-legacy --drop-edge 5-8 3-2
