#!/bin/bash	

# Mask only experimentis
python eval_annotation_method.py --policy qnet_mask --db DAVIS_17
python eval_annotation_method.py --policy oracle_mask --db DAVIS_17
python eval_annotation_method.py --policy rand_mask --db DAVIS_17
python eval_annotation_method.py --policy l2_mask --encoder dino_large --db DAVIS_17
python eval_annotation_method.py --policy l2_mask --encoder resnet301 --db DAVIS_17
python eval_annotation_method.py --policy l2_mask --encoder vit_large --db DAVIS_17

# Multiple annotation types experiments
python eval_annotation_method.py --policy oracle_oracle --types 3clicks mask --db DAVIS_17
python eval_annotation_method.py --policy eva_vos --types 3clicks mask --db DAVIS_17 
python eval_annotation_method.py --policy rand_rand --types 3clicks mask --db DAVIS_17
python eval_annotation_method.py --policy rand_type --types 3clicks --db DAVIS_17