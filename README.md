#


```bash



POINT_PATH=/home/zhihao/Publications/output/gaussian/FLAME_zhang_1111_time_EMO-1/point_cloud/iteration_100000/point_cloud.ply


SEGMENT_PATH=/home/zhihao/Publications/data/MRI/MRI_zhang/zhang
TRANS_PATH=/home/zhihao/Publications/output/alignment/transform/alignment_transform.npz

python local_viewer.py  \
--point-path ${POINT_PATH} \
--segment-path ${SEGMENT_PATH} \
--transform-path ${TRANS_PATH} \
--lbs \
--skull-jaw 4 5 \


```