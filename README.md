# GaussianAvatars

## require

```bash
conda activate gaussian-avatars
export CUDA_VISIBLE_DEVICES=1
```

## GA

```bash

SUBJECT=luo_1104
SEQUENCE=EMO-1
ITER=100000
OUTPUT=Photo_${SUBJECT}_${SEQUENCE}_50

# training
python train.py \
-s ../output/export/${SUBJECT}_${SEQUENCE} \
-m ../output/gaussian/${OUTPUT} \
--iterations ${ITER} --interval 10000 \
--eval --bind_to_mesh --white_background --port 60001 --points_per_face 50

# during training
python remote_viewer.py --port 60000

# after training
python local_viewer.py  --point_path ../output/gaussian/${OUTPUT}/point_cloud/iteration_${ITER}/point_cloud.ply 

mkdir -p ./output/gaussian/${OUTPUT}/point_cloud/
scp -r cuigpu01:/home/zhangzhh12024/Avatars/output/gaussian/${OUTPUT}/point_cloud/iteration_${ITER} ./output/gaussian/${OUTPUT}/point_cloud/iteration_${ITER}

```
<!-- =========================================================================================== -->
## Using mri fitting

### fitting

```bash
conda activate py36
SUBJECT="zhang_128"
SEQUENCE="EMO-1" 
MESH_PATH='../data/MRI/MRI_zhang/zhang_downsampled/Segment_20.obj'
LMK_PATH='../data/MRI/MRI_zhang/zhang_downsampled/zhang_51l.pp'

SUBJECT="luo_128"
SEQUENCE="EMO-1" 
MESH_PATH='../data/MRI/MRI_luo/luo_downsampled/Segment_20.obj'
LMK_PATH='../data/MRI/MRI_luo/luo_downsampled/luo_51.pp'

python MRI_fitting.py -s ../output/export/${SUBJECT}_${SEQUENCE} --scan_path ${MESH_PATH} --lmk_path ${LMK_PATH}

```

```bash
SUBJECT="zhang_128" 
SEQUENCE="EMO-1" 
ITER=100000
OUTPUT="MRI_${SUBJECT}_${SEQUENCE}"


SUBJECT="luo_128" 
SEQUENCE="EMO-1" 
ITER=100000
OUTPUT="MRI_${SUBJECT}_${SEQUENCE}"
```

```bash
MESH_PATH="../output/export/${SUBJECT}_${SEQUENCE}/fitting_results/stage3_fitted_flame.obj"
# train
python train.py \
-s ../output/export/${SUBJECT}_${SEQUENCE} \
-m ../output/gaussian/${OUTPUT} \
--iterations ${ITER} --interval 10000 \
--eval --bind_to_mesh --white_background \
--use_mri_model  --mesh_path ${MESH_PATH} --port 60000

# during training
python remote_viewer.py --port 60000


SEGM_PATH='../data/MRI/MRI_zhang/MRI_zhangzhihao-pd-mx3d-sag-iso0.6mm-non-fat-201628-401_all.nii.gz-models'
# after training
python local_viewer.py --use_mri_model \
--point_path ../output/gaussian/${OUTPUT}/point_cloud/iteration_${ITER}/point_cloud.ply --mesh_path ${MESH_PATH} --use_segm --segm_path ${SEGM_PATH}


SUBJECT="luo_128" 
SEQUENCE="EMO-1" 
OUTPUT="MRI_${SUBJECT}_${SEQUENCE}"
ITER=50000
MESH_PATH="/home/zhihao/NeRSemble/output/export/luo_128_EMO-1/fitting_results/stage3_fitted_flame.obj"
SEGM_PATH="../data/MRI/MRI_luo/luo_downsampled"


SUBJECT="zhang_128" 
SEQUENCE="EMO-1" 
ITER=100000
OUTPUT="MRI_${SUBJECT}_${SEQUENCE}"
MESH_PATH="../output/export/${SUBJECT}_${SEQUENCE}/fitting_results/stage3_fitted_flame.obj"
SEGM_PATH="../data/MRI/MRI_zhang/zhang_downsampled"


python local_viewer_flame.py \
--point_path ../output/gaussian/${OUTPUT}/point_cloud/iteration_${ITER}/point_cloud.ply \
--use-hybrid-model --mesh_path ${MESH_PATH}  \
--use_segm --segm_path ${SEGM_PATH} --jaw_id 5


python local_viewer_lbs.py \
--point_path ../output/gaussian/${OUTPUT}/point_cloud/iteration_${ITER}/point_cloud.ply \
--use_mri_model --mesh_path ${MESH_PATH} \
--use_segm --segm_path ${SEGM_PATH} --skull_jaw 4 5
```

## msic

```Bash
REMOTE=

scp -P 22112 -r zhangzhh12024@10.15.49.36:"${REMOTE/#\/home/\/grpczm}" .
```

$(seq 10000 10000 $ITER | tr '\n' ' ')

SUBJECT="luo_128"
SEQUENCE="EMO-1"
ITER=10000

MESH_PATH='/home/zhihao/NeRSemble/data/MRI/fit_scan_result_70.obj'

SUBJECT="luo_128"
SEQUENCE="EMO-1"
ITER=10000
OUTPUT="MRI_head"
MESH_PATH='/home/zhihao/NeRSemble/data/MRI/MRI_luotao_skin_rot.obj'

SUBJECT="luo_128"
SEQUENCE="EMO-1"
ITER=10000
OUTPUT="MRI_head-0620_fit70"
MESH_PATH="/home/zhihao/NeRSemble/data/MRI/merge_obj_result.obj"

SUBJECT="luo_128"
SEQUENCE="rigid"
ITER=10000
OUTPUT="MRI_head_fit70"
MESH_PATH='../data/MRI/MRI_luotao_fit_scan_70lmk.obj'
