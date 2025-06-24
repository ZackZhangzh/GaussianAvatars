#

## GaussianAvatars

```bash

conda activate gaussian-avatars
export CUDA_VISIBLE_DEVICES=1
```

```bash
# train
python train.py \
-s ../output/export/${SUBJECT}_${SEQUENCE} \
-m ../output/gaussian/${OUTPUT} \
--iterations ${ITER} --interval 10000 \
--eval --bind_to_mesh --white_background \
--use_mri_model  --mesh_path ${MESH_PATH} \
--port 60000

# during training
python remote_viewer.py --port 60000

# after training
python local_viewer.py --use_mri_model \
--point_path ../output/gaussian/${OUTPUT}/point_cloud/iteration_${ITER}/point_cloud.ply --mesh_path ${MESH_PATH}
```

$(seq 10000 10000 $ITER | tr '\n' ' ')

## cfg

```bash
export CUDA_VISIBLE_DEVICES=5
SUBJECT="luo_128"
SEQUENCE="rigid" 
ITER=300000
OUTPUT="MRI_fit_70lmk"
MESH_PATH='../data/MRI/MRI_luotao_fit_scan_70lmk.obj'


export CUDA_VISIBLE_DEVICES=6
SUBJECT="luo_128"
SEQUENCE="rigid" 
ITER=300000
OUTPUT="MRI_merge"
MESH_PATH="../data/MRI/MRI_luotao_merge.obj"

export CUDA_VISIBLE_DEVICES=7
SUBJECT="luo_128"
SEQUENCE="rigid" 
ITER=300000
OUTPUT="MRI_original"
MESH_PATH="../data/MRI/MRI_luotao_skin_original.obj"




```

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
