#!/bin/sh
SUBJECT=zhang_1111
SEQUENCE=EMO-1

OUTPUT=GA_${SUBJECT}_${SEQUENCE}
ITER=100000

cd /home/zhangzhh12024/Avatars/GaussianAvatars
conda activate gaussian-avatars
export CUDA_VISIBLE_DEVICES=1
# training
python train.py \
-s ../output/export/${SUBJECT}_${SEQUENCE} \
-m ../output/gaussian/${OUTPUT} \
--iterations ${ITER} --interval 10000 \
--eval --bind_to_mesh --white_background \
--port 60000 \
--points_per_face 5

# viewer
python remote_viewer.py --port 60000


python local_viewer.py  --point_path ../output/gaussian/${OUTPUT}/point_cloud/iteration_${ITER}/point_cloud.ply 

scp -r cuigpu01:/home/zhangzhh12024/Avatars/output/gaussian/${OUTPUT}/point_cloud/iteration_${ITER} /home/zhihao/Avatars/output/gaussian/${OUTPUT}/point_cloud/

# # # # # # # # # # # # 


OUTPUT="MRI_${SUBJECT}_${SEQUENCE}_x50"
ITER=100000

MESH_PATH=/home/zhangzhh12024/Avatars/data/MRI/MRI_zhang/user_mesh_aligned.obj

python train.py \
-s ../output/export/${SUBJECT}_${SEQUENCE} \
-m ../output/gaussian/${OUTPUT} \
--iterations ${ITER} --interval 10000 \
--eval --bind_to_mesh --white_background \
--port 60001 \
--points_per_face 50 \
--use_mri_model  --mesh_path ${MESH_PATH}

####




python local_viewer.py  --point_path  /home/zhangzhh12024/Avatars/output/gaussian/MRI_zhang_1111_EMO-1_x10/point_cloud/iteration_100000/point_cloud.ply --use_mri_model  --mesh_path ${MESH_PATH}