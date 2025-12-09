#


```bash
PROJECT_ROOT=${HOME}/Publications/
conda activate gaussian-avatars 

MESH_PATH=${PROJECT_ROOT}data/MRI/MRI_zhang/zhang/Segment_20.obj
LMK_PATH=${PROJECT_ROOT}output/landmarks/landmarks.pp
python mesh_align_to_flame.py \
--flame-reference ${PROJECT_ROOT}output/export/zhang_1111_time_EMO-1 \
--mesh-path ${MESH_PATH} \
--lmk-path ${LMK_PATH} \
--output-dir ${PROJECT_ROOT}output/alignment \
--landmark-type static  --no-enable-scaling 







POINT_PATH=${PROJECT_ROOT}output/gaussian/FLAME_zhang_1111_time_EMO-1/point_cloud/iteration_100000/point_cloud.ply
SEGMENT_PATH=${PROJECT_ROOT}data/MRI/MRI_zhang/zhang
TRANS_PATH=${PROJECT_ROOT}output/alignment/transform/alignment_transform.npz

python local_viewer.py  \
--point-path ${POINT_PATH} \
--segment-path ${SEGMENT_PATH} \
--transform-path ${TRANS_PATH} \
--lbs \
--skull-jaw 4 5 \
--debug

```