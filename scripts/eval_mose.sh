#!/bin/bash	

python download_data.py
# Mask only experimentis
python eval_annotation_method.py --policy qnet_mask
python eval_annotation_method.py --policy oracle_mask
python eval_annotation_method.py --policy rand_mask
python eval_annotation_method.py --policy l2_mask --encoder dino_large
python eval_annotation_method.py --policy l2_mask --encoder resnet101
python eval_annotation_method.py --policy l2_mask --encoder vit_large

# Multiple annotation types experiments
python eval_annotation_method.py --policy oracle_oracle --types 3clicks mask
python eval_annotation_method.py --policy eva_vos --types 3clicks mask
python eval_annotation_method.py --policy rand_rand --types 3clicks mask
python eval_annotation_method.py --policy rand_type --types 3clicks