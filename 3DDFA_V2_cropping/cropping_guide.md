Cropping and obtaining camera poses guide using [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2).

Our cropping file [recrop_images.py](./recrop_images.py) is based on 3DDFA_V2 and dlib. First of all, please clone the 3DDFA_V2 repo and follow all the installation instruction and make sure their demos can be run sucessfully. After building cython of 3DDFA, you can use the follow command for other necessary packages:

```
pip install opencv-python dlib pyyaml onnxruntime onnx
```

Test images used here are from [test/origin/man.jpg](https://www.freepik.com/free-photo/portrait-white-man-isolated_3199590.htm) and [test/origin/woman.jpg](https://www.freepik.com/free-photo/pretty-smiling-joyfully-female-with-fair-hair-dressed-casually-looking-with-satisfaction_9117255.htm)

---

# Steps

## 1. Move folder `test`, `recrop_images.py`, and `dlib_kps.py` under this directory to the 3DDFA_V2 root dir. Also, remember to download `shape_predictor_68_face_landmarks.dat` and put it under 3DDFA_V2 root dir.

## 2. cd to 3DDFA_V2. The cropping script has to run under 3DDFA_V2 dir.
```.bash
cd 3DDFA_V2
```

## 3. Extract face keypoints using dlib. After this, you should have data.pkl under your 3DDFA root dir saving the keypoints.
```.bash
python dlib_kps.py 
```

## 4. Obtaining camera poses and cropping the images using recrop_images.py

```.bash
python recrop_images.py -i data.pkl -j dataset.json
```

## After this, you should have a folder called crop_samples in your 3DDFA_V2 root dir, which is the same as the one under this directory. Then, you can move the folder back to our panohead root dir, change the --target_img flag to the corresponding folder in `gen_pti_script.sh`, and do the PTI inversion.