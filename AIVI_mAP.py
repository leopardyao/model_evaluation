import os
import sys
import shutil
import yaml
from lib.detection import Detection


# sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))
# from lib.detection import detections, plot_save_result

def reset(Path):
    if os.path.exists(Path):  # if it exist already
        # reset the output directory
        shutil.rmtree(Path)  # tt: remove , and establish current
    os.makedirs(Path)


conf_path = './conf/conf.yaml'
with open(conf_path, 'r', encoding='utf-8') as f:
    data = f.read()
cfg = yaml.safe_load(data)

model_dir = '/home/usr/work_unix/IBM/vi_edge/pred/vi_edge_autoline6/'
#project_dir = '/home/usr/work_unix/IBM/py-faster-rcnn_AX/output/faster_rcnn_alt_opt/voc_2007_test/VGG16_faster_rcnn_final_scratchmix/'
data_dir = 'test_nooverkillandcrack'
# project_dir = 'C:\\Users\\tangjny\\Station\\work_station\\Software\\Tools\\model-evaluation\\'
# model_dir = 'data'
# 根据config文件里的dataformat, 如果是读txt文件，我会从下面两个文件夹读，如果是读csv, 我也会从model_dir读这两个csv 文件
gtFolder = os.path.join(model_dir, data_dir, 'ground-truth')
detFolder = os.path.join(model_dir, data_dir, 'detection-results')
# 模型的输出结果，
savePath = os.path.join(model_dir, data_dir, 'evaluation-results')

reset(savePath)

detection = Detection(cfg, gtFolder, detFolder, savePath)
detection.cal_mAP()
detection.plot_save_result()
