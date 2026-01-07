#

## Training

```bash
SUBJECT=rigid_zhang_1111
SEQUENCE=EMO-1
MESH_NAME=user_mesh_aligned_decimated_0.5
MESH_PATH=${DATA}/Avatars/Model/facescan_zhang/${MESH_NAME}.obj
ITER=100000
OUTPUT_NAME=facescan_${SUBJECT}_${SEQUENCE}_${MESH_NAME}
# MESH_PATH=${PROJECT_ROOT}/data/MRI/MRI_zhang/user_mesh_aligned.obj
# MESH_PATH=${PROJECT_ROOT}/data/MRI/facescan/user_mesh_aligned_decimated_8861.obj


# train
python train.py \
-s ${DATA}/output/export/${SUBJECT}_${SEQUENCE} \
-m ${DATA}/output/gaussian/${OUTPUT_NAME} \
--iterations ${ITER} --interval 10000 \
--eval --bind_to_mesh --white_background \
--mesh_path ${MESH_PATH} --port 60000 \
--optimize_mesh_transform \
# --mesh_debug_interval 1000 \
# --init_mesh_transform ${DATA}/Avatars/Model/facescan_zhang/mesh_param.npz


python train.py \
-s ${DATA}/output/export/${SUBJECT}_${SEQUENCE} \
-m ${DATA}/output/gaussian/${OUTPUT_NAME}_x10 \
--iterations ${ITER} --interval 10000 \
--eval --bind_to_mesh --white_background \
--mesh_path ${MESH_PATH} --port 60002 \
--optimize_mesh_transform --point_per_face 10  \
# --init_mesh_transform ${DATA}/Avatars/Model/facescan_zhang/mesh_param.npz


# optional
--mesh_debug_interval 1000 \
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
# POINT_PATH=/data/Data/Projects_data/output/gaussian/facescan_zhang_1111_EMO-1_user_mesh_aligned_decimated_0.1/point_cloud/iteration_1000/point_cloud.ply



# MESH_PATH=${DATA}/Avatars/Model/facescan_zhang/user_mesh_aligned_decimated_0.1.obj
# POINT_PATH=${PROJECT_ROOT}/output/gaussian/rigid_MESH_rigid_zhang_1111_EMO-1/iteration_10000/point_cloud.ply
# python local_viewer.py \
# --point-path ${POINT_PATH} \
# --mesh-path ${MESH_PATH}

SUBJECT=rigid_zhang_1111
SEQUENCE=EMO-1
MESH_NAME=user_mesh_aligned_decimated_0.5
MESH_PATH=${DATA}/Avatars/Model/facescan_zhang/${MESH_NAME}.obj
POINT_PATH=$DATA/output/gaussian/facescan_rigid_zhang_1111_EMO-1_user_mesh_aligned_decimated_0.5_x10/point_cloud/iteration_30000/point_cloud.ply  


MESH_NAME=user_mesh_aligned_decimated_0.2
MESH_PATH=${DATA}/Avatars/Model/facescan_zhang/${MESH_NAME}.obj
POINT_PATH=$DATA/output/gaussian/facescan_rigid_zhang_1111_EMO-1_user_mesh_aligned_decimated_0.2

# MESH_NAME=user_mesh_aligned_decimated_0.2
# MESH_PATH=${DATA}/Avatars/Model/facescan_zhang/${MESH_NAME}.obj
# POINT_PATH=$DATA/output/gaussian/facescan_rigid_zhang_1111_EMO-1_user_mesh_aligned_decimated_0.2_x50


python local_viewer.py \
--point-path ${POINT_PATH}  \
--mesh-path ${MESH_PATH}



SEGMENT_PATH=${HOME}/Publications/data/MRI/MRI
TRANS_PATH=$DATA/Avatars/Model/MRI_zhang/alignment_transform.npz 
python local_viewer.py \
--point-path ${POINT_PATH} \
--mesh-path ${MESH_PATH} \
--segment-path ${SEGMENT_PATH} \
--transform-path ${TRANS_PATH} \
--lbs joint  \
--skull-jaw 4 5 \
--debug


MESH_PATH=${DATA}/Avatars/Model/facescan_zhang/user_mesh_aligned.obj
POINT_PATH=/home/zhihao/Publications/output/gaussian/rigid_MESH_rigid_zhang_1111_EMO-1/iteration_100000/point_cloud.ply
```
<!-- 
<!--  -->
<!-- 
 <!--
  <!--
   <!--
  <!--
  <!--
   <!-- <!--
  <!--
   <!-- <!--
  <!--
   <!--
  -->
## Mesh Alignment

```bash
PROJECT_ROOT=${HOME}/Publications/
conda activate gaussian-avatars 

MESH_PATH=/home/zhihao/Publications/data/LightStage/17.obj
LMK_PATH=/home/zhihao/Publications/output/landmarks/landmarks_3d_68.npy

MESH_PATH=/home/zhihao/Publications/data/MRI/MRI/Segment_20.obj
LMK_PATH=/home/zhihao/Publications/data/MRI/MRI/landmarks_3d_51.npy

python mesh_align_to_flame.py \
--source-mesh $MESH_PATH \
--source-lmk $LMK_PATH \
--target-mode custom \
--target-mesh $DATA/Avatars/Model/facescan_zhang/user_mesh_aligned.obj \
--target-lmk $DATA/Avatars/Model/facescan_zhang/user_landmarks_aligned_68.npy \
--output-dir ./output/alignment \
--landmark-type static --enable-scaling \
--manual-fine-tune 




python mesh_align_to_flame.py \
--flame-reference ${PROJECT_ROOT}/output/export/zhang_1111_time_EMO-1 \
--mesh-path ${MESH_PATH} \
--lmk-path ${LMK_PATH} \
--output-dir ${PROJECT_ROOT}/output/alignment \
--landmark-type full  --enable-scaling \
--manual-fine-tune #可选 --load-alignment-path ${TRANS_PATH}




python visualize_camera_mesh.py \
--source-path ${PROJECT_ROOT}/output/export/${SUBJECT}_${SEQUENCE}  \
--mesh-path ${MESH_PATH} \
--vis_cam 1

```
