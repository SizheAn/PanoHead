#!/usr/bin/env bash

models=("easy-khair-180-gpc0.8-trans10-025000.pkl"\
  "ablation-trigridD-1-025000.pkl"\
)


for model in ${models[@]}

do 


python calc_metrics.py --network models/${model} \
--img_data=dataset/ffhq-3DDFA-exp-augx2-lpx2-easyx2-khairfiltered-img.zip\
 --seg_data=dataset/ffhq-3DDFA-exp-augx2-lpx2-easyx2-khairfiltered-seg.zip\
  --gpus 8 --mirror True --metrics fid50k_full,is50k | tee -a paper_metrics/${model}.log


done
