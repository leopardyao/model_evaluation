import os
import sys
from collections import Counter
import time

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display


class Evaluator:
    def GetPascalVOCMetrics(self,
                            cfg,
                            classes, # tt: ground truth classes
                            gt_boxes,  # tt format: {'class1':{'imgID':[[x1,y1,x2,y2,0],[...]],'imgID2:...},...}
                            num_pos,  # format:{'class1':#,'class2',#}
                            det_boxes):  # format:{'class1':[[x1,x2,y1,y2,confidence,imgID],[...]],'class2':[[..]]}

        ret = []
        # groundTruths = []
        # detections = []

        for c in classes:
            if c not in det_boxes.keys():  # if ground truth has , but detection not have
                r = {
                    'class': c,
                    'precision': [0],
                    'recall': [0],
                    'confidence':[0],
                    'AP': 0,
                    'interpolated precision': [0],
                    'interpolated recall': [0],
                    'total positives': 0,
                    'total TP': 0,
                    'total FP': 0,
                }
                ret.append(r)
                continue
            else:
                dects = det_boxes[c]
            gt_class = gt_boxes[c]  # still a dictonary
            npos = num_pos[c]
            dects = sorted(dects, key=lambda conf: conf[4], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            CONFIDENCE = np.zeros(len(dects))
            for d in range(len(dects)):
                CONFIDENCE[d]= dects[d][4]
                iouMax = sys.float_info.min
                if dects[d][-1] in gt_class:  # here check if the imageID is in gt_class
                    for j in range(len(gt_class[dects[d][-1]])):  # tt: 循环每一个gtbox,找到iou max
                        iou = Evaluator.iou(dects[d][:4], gt_class[dects[d][-1]][j][:4])
                        if iou > iouMax:
                            iouMax = iou
                            jmax = j

                    if iouMax >= cfg['iouThreshold']:  # for image 943b2e12, IOU was only 0.4, although is correct
                        if gt_class[dects[d][-1]][jmax][4] == 0:  # tt :没有被匹配过
                            TP[d] = 1
                            gt_class[dects[d][-1]][jmax][4] == 1

                        else:
                            FP[d] = 1
                    else:
                        FP[d] = 1
                else:
                    FP[d] = 1

            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))

            [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            AP_str = "{0:.2f}%".format(ap * 100)
            # print('mAP: %s' % mAP_str)
            print('AP for %s = %s' % (c, AP_str))
            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'confidence': CONFIDENCE,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP),
            }
            ret.append(r)
        return ret

    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[i + 1] != mrec[i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    # tt this is the function I borrowed from simplerfastrcnn project
    def calc_detection_voc_ap(prec, rec, use_07_metric=False):
        """Calculate average precisions based on evaluation code of PASCAL VOC.

        This function calculates average precisions
        from given precisions and recalls.
        The code is based on the evaluation code used in PASCAL VOC Challenge.

        Args:
            prec (list of numpy.array): A list of arrays.
                :obj:`prec[l]` indicates precision for class :math:`l`.
                If :obj:`prec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            rec (list of numpy.array): A list of arrays.
                :obj:`rec[l]` indicates recall for class :math:`l`.
                If :obj:`rec[l]` is :obj:`None`, this function returns
                :obj:`numpy.nan` for class :math:`l`.
            use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
                for calculating average precision. The default value is
                :obj:`False`.

        Returns:
            ~numpy.ndarray:
            This function returns an array of average precisions.
            The :math:`l`-th value corresponds to the average precision
            for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
            :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.

        """

        n_fg_class = len(prec)
        ap = np.empty(n_fg_class)
        for l in six.moves.range(n_fg_class):
            if prec[l] is None or rec[l] is None:
                ap[l] = np.nan
                continue

            if use_07_metric:
                # 11 point metric
                ap[l] = 0
                for t in np.arange(0., 1.1, 0.1):
                    if np.sum(rec[l] >= t) == 0:
                        p = 0
                    else:
                        p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                    ap[l] += p / 11
            else:
                # correct AP calculation
                # first append sentinel values at the end
                mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
                mrec = np.concatenate(([0], rec[l], [1]))

                mpre = np.maximum.accumulate(mpre[::-1])[::-1]

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB)
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        if iou < 0:
            import pdb
            pdb.set_trace()
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
