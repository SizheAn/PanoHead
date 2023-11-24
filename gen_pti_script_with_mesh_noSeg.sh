#!/usr/bin/env bash

model="easy-khair-180-gpc0.8-trans10-025000.pkl"

input_dir="models"
output_dir="pti_out"

target_img="dataset/testdata_img"
#target_seg="dataset/testdata_seg"


# Perform the pti and save w
python projector_withseg.py --outdir="${output_dir}" --target_img="${target_img}" --network "${input_dir}/${model}" --idx "0" --shapes=True --save-video=False
# Generate .mp4 before finetune
python gen_videos_proj_withseg.py --output="${output_dir}/${model}/0/PTI_render/pre.mp4" --latent="${output_dir}/${model}/0/projected_w.npz" --trunc 0.7 --network "${input_dir}/${model}" --cfg Head
# Generate .mp4, .ply mesh and frame images after finetune
    python gen_videos_proj_withseg.py --output="${output_dir}/${model}/0/PTI_render/post.mp4" --latent="${output_dir}/${model}/0/projected_w.npz" --trunc 0.7 --network "${output_dir}/${model}/0/fintuned_generator.pkl" --cfg Head --shapes True --frames True --level 42

done
