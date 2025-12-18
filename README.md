#

## Mesh Alignment

```bash
PROJECT_ROOT=${HOME}/Publications/
conda activate gaussian-avatars 

MESH_PATH=/home/zhihao/Publications/data/LightStage/17.obj
LMK_PATH=/home/zhihao/Publications/output/landmarks/landmarks_3d_68.npy


python mesh_align_to_flame.py \
--flame-reference ${PROJECT_ROOT}/output/export/zhang_1111_time_EMO-1 \
--mesh-path ${MESH_PATH} \
--lmk-path ${LMK_PATH} \
--output-dir ${PROJECT_ROOT}/output/alignment \
--landmark-type full  --enable-scaling \
--manual-fine-tune #可选 --load-alignment-path ${TRANS_PATH}
```

## Training

```bash
PROJECT_ROOT=${HOME}/Publications/

SUBJECT=zhang_1111
SEQUENCE=EMO-1
OUTPUT=MESH_${SUBJECT}_${SEQUENCE}
ITER=100000

MESH_PATH=/home/zhihao/Publications/data/MRI/MRI_zhang/Segment_20_decimated.obj


# train
python train.py \
-s ${PROJECT_ROOT}/output/export/${SUBJECT}_${SEQUENCE} \
-m ${PROJECT_ROOT}/output/gaussian/${OUTPUT} \
--iterations ${ITER} --interval 10000 \
--eval --bind_to_mesh --white_background \
--mesh_path ${MESH_PATH} --port 60000 \
--optimize_mesh_transform --mesh_debug_interval 100 \
--mesh_pose_lr 1e-3 --mesh_trans_lr 1e-3



PROJECT_ROOT=${HOME}/Avatars
SUBJECT=rigid_zhang_1111
SEQUENCE=EMO-1
OUTPUT=rigid_MESH_${SUBJECT}_${SEQUENCE}
ITER=100000
MESH_PATH=${PROJECT_ROOT}/data/MRI/facescan/user_mesh_aligned.obj




```

## Visualization

```bash
POINT_PATH=${PROJECT_ROOT}output/gaussian/MESH_zhang_1111_time_EMO-1/point_cloud/iteration_100000/point_cloud.ply
SEGMENT_PATH=${PROJECT_ROOT}data/MRI/MRI_zhang/zhang
TRANS_PATH=${PROJECT_ROOT}output/alignment/transform/alignment_transform.npz


python local_viewer.py  \
--point-path ${POINT_PATH} \
--segment-path ${SEGMENT_PATH} \
--transform-path ${TRANS_PATH} \
--lbs \
--skull-jaw 4 5 \
--debug




POINT_PATH=${PROJECT_ROOT}/output/gaussian/MRI_zhang_1111_time_EMO-1/point_cloud/iteration_10000/point_cloud.ply
python local_viewer.py  \
--point-path ${POINT_PATH} --segment-path ${SEGMENT_PATH}
```
