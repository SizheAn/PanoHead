from codecs import ignore_errors
from genericpath import isdir
import sys
from subprocess import call
import os
import torch
import random
import string

import shutil

os.environ['KMP_DUPLICATE_LIB_OK']='True'


path_3DDFA = "E:/3DDFA_V2-master/"
path_crop_targets = "test/original/"
path_crop_results = "crop_samples/"
path_target_img = "dataset/testdata_img/"

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except Exception as e:
        print(f"Error: {e}!")


import gradio as gr



def inference (img):    
    filename = os.path.basename(img)    
    serialname = f''.join(random.choices(string.digits, k=6))
    os.chdir(path_3DDFA)
    
    #flush
    shutil.rmtree(path_crop_results, ignore_errors= True)
    shutil.rmtree(path_3DDFA + path_crop_targets, ignore_errors= True)    
    os.makedirs(f'./' + path_crop_targets, exist_ok = True)
    #copy img
    shutil.copyfile(img, path_3DDFA + path_crop_targets + serialname + os.path.splitext(filename)[1])
    
    run_cmd(f'python dlib_kps.py')
    run_cmd(f'python recrop_images.py -i data.pkl -j dataset.json')
    if os.path.isfile( path_3DDFA + path_crop_results + serialname + f'.jpg' ) == False:
        print("Error: no face")
        return
    os.chdir(os.path.dirname(__file__)) #return to python file path
    #os.chdir(os.path.dirname(os.getcwd())) #return to root    


    #flush2
    shutil.rmtree(path_target_img, ignore_errors= True)
    #os.makedirs(f'./' + path_target_img, exist_ok = True)
    #copy2
    shutil.copytree(path_3DDFA + path_crop_results , path_target_img)

    run_cmd(f'gen_pti_script_with_mesh_noSeg.sh')

    #store .ply and post.mp4 separately under a new folder
    if os.isdir('out') == False:
        os.mkdir('out')
    os.mkdir(f'out/' + serialname)
    
    shutil.copyfile('pti_out/easy-khair-180-gpc0.8-trans10-025000.pkl/0/PTI_render/post.mp4', f'out/' + serialname + '/post.mp4')
    shutil.copyfile('pti_out/easy-khair-180-gpc0.8-trans10-025000.pkl/0/PTI_render/post.ply', f'out/' + serialname + '/post.ply')
    shutil.copyfile(path_target_img + serialname + '.jpg', f'out/' + serialname + '/'+ serialname + '.jpg' )
    shutil.copyfile(path_target_img + 'dataset.json', f'out/' + serialname + '/dataset.json' )
    
    return f'pti_out/easy-khair-180-gpc0.8-trans10-025000.pkl/0/PTI_render/post.mp4'


title= "Panohead demo"
description= "Panohead demo for gradio"
article= "<p style='text-align: center'><a href='https://github.com/SizheAn/PanoHead'>Github Repo</a></p>"
examples= [       
]

demo = gr.Interface(
     inference,     
     gr.Image(type="filepath"), 
     outputs=gr.Video(label="Out"),
    title=title,
    description=description,
    article=article,
    examples=examples
)

demo.launch()
