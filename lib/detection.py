import os
import matplotlib.pyplot as plt
from lib.Evaluator import *
import pdb
import pandas as pd
import operator
import numpy as np


class Detection:

    def __init__(self, cfg, gtFolder, detFolder, savePath):
        # super(RCNNTest, self).__init__()
        self.cfg = cfg
        self.GTFolder = gtFolder
        self.DetFolder = detFolder
        self.savePath = savePath
        self.df_gt = None
        self.num_pos = None
        self.df_dr = None
        self.num_det = None

        # detection results####
        self.results = None
        self.gt_classes = None
        self.det_classes = None

    def getGTBoxes(self):
        classes = []
        num_pos = {}
        gt_boxes = {}  # {'class':{'imgID'[[x1,y1,x2,y2,0],[...]}}

        if self.cfg['dataformat'] == 'txt':
            files = os.listdir(self.GTFolder)
            files.sort()
            for f in files:
                nameOfImage = f.replace(".txt", "")
                fh1 = open(os.path.join(self.GTFolder, f), "r")

                for line in fh1:
                    line = line.replace("\n", "")
                    if line.replace(' ', '') == '':
                        continue
                    splitLine = line.split(" ")

                    cls = (splitLine[0])  # class
                    left = float(splitLine[1])
                    top = float(splitLine[2])
                    right = float(splitLine[3])
                    bottom = float(splitLine[4])
                    one_box = [left, top, right, bottom, 0]  # tt: 最后一个0在后面表示是否有匹配到

                    if cls not in classes:
                        classes.append(cls)
                        gt_boxes[cls] = {}
                        num_pos[cls] = 0

                    num_pos[cls] += 1

                    if nameOfImage not in gt_boxes[cls]:
                        gt_boxes[cls][nameOfImage] = []
                    gt_boxes[cls][nameOfImage].append(one_box)

                fh1.close()
        else:
            file_csv = self.GTFolder + '.csv'
            self.df_gt = pd.read_csv(file_csv)
            # num_imgs = self.df_gt['imgID'].nunique()
            classes = list(self.df_gt['class_name'].unique())
            self.num_pos = self.df_gt['class_name'].value_counts()
            # below I convert the dataframe to original source code evaluator required format:
            # {'class':{'imgID'[[x1,y1,x2,y2,0],[...]}}
            self.df_gt['flag'] = 0
            self.df_gt['bbox'] = self.df_gt[['xmin', 'ymin', 'xmax', 'ymax', 'flag']].values.tolist()
            dfg = self.df_gt.groupby(['class_name', 'imgID'])['bbox'].apply(list)
            # convert multi-index series to nested dict to fit with evaluator format
            gt_boxes = {level: dfg.xs(level).to_dict() for level in dfg.index.levels[0]}

        return gt_boxes, classes, self.num_pos

    def getDetBoxes(self):
        det_boxes = {}

        if self.cfg['dataformat'] == 'txt':
            files = os.listdir(self.DetFolder)
            files.sort()
            for f in files:
                nameOfImage = f.replace(".txt", "")
                fh1 = open(os.path.join(self.DetFolder, f), "r")

                for line in fh1:
                    line = line.replace("\n", "")
                    if line.replace(' ', '') == '':
                        continue
                    splitLine = line.split(" ")

                    cls = (splitLine[0])  # class
                    left = float(splitLine[2])
                    top = float(splitLine[3])
                    right = float(splitLine[4])
                    bottom = float(splitLine[5])
                    score = float(splitLine[1])
                    one_box = [left, top, right, bottom, score, nameOfImage]

                    if cls not in det_boxes:
                        det_boxes[cls] = []
                    det_boxes[cls].append(one_box)

                fh1.close()
        else:
            file_csv = self.DetFolder + '.csv'
            self.df_dr = pd.read_csv(file_csv)  # detection results
            self.df_dr['confidence'] = self.df_dr['confidence'].round(3)
            self.df_dr['bbox'] = self.df_dr[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'imgID']].values.tolist()
            self.num_det = self.df_dr['class_name'].value_counts()
            dfg = self.df_dr.groupby(['class_name'])['bbox'].apply(list)
            # convert multi-index series to nested dict to fit with evaluator format
            det_boxes = dfg.to_dict()

        det_classes = list(self.df_dr['class_name'].unique())

        return det_boxes, det_classes

    def cal_mAP(self):
        gt_boxes, self.gt_classes, num_pos = self.getGTBoxes()
        det_boxes, self.det_classes = self.getDetBoxes()

        evaluator = Evaluator()
        self.results = evaluator.GetPascalVOCMetrics(self.cfg, self.gt_classes, gt_boxes, num_pos, det_boxes)

        return self.results

    def plot_save_result(self, to_show=False):
        num_imgs = 0
        if self.df_gt is not None:
            num_imgs = self.df_gt['imgID'].nunique()
        num_classes = len(self.gt_classes)
        plt.rcParams['savefig.dpi'] = 80
        plt.rcParams['figure.dpi'] = 130

        acc_AP = 0
        validClasses = 0
        fig_index = 0
        # 1. plot ground truth
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(num_imgs) + " files and " + str(num_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = self.savePath + "/ground-truth-info.png"
        plot_color = 'forestgreen'
        self.draw_plot_func(
            self.num_pos,
            num_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
        )

        ap_dict = {}  # average precision dictionary, for later plotting purpose
        tp_dict = {}  # true posistive dictonary , for later plotting purposes
        for cls_index, result in enumerate(self.results):
            if result is None:
                raise IOError('Error: Class %d could not be found.' % cls_index)

            cls = result['class']
            precision = result['precision']
            recall = result['recall']
            confid = result['confidence']
            average_precision = result['AP']
            ap_dict[cls] = average_precision
            acc_AP = acc_AP + average_precision
            mpre = result['interpolated precision']
            mrec = result['interpolated recall']
            npos = result['total positives']
            total_tp = result['total TP']
            total_fp = result['total FP']

            tp_dict[cls] = total_tp
            # get index of points I want to display confidence
            confid_index = np.linspace(0, len(precision), num=5, endpoint=False, dtype=int)
            # fig_index+=1
            plt.figure()
            plt.plot(recall, precision, self.cfg['colors'][cls_index], label='Precision')
            for index in confid_index:
                confid_label = "{:.2f}".format(confid[index])
                plt.annotate(confid_label,  # this is the text
                             (recall[index], precision[index]),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             arrowprops=dict(facecolor='black', arrowstyle='-'),
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center
            plt.xlabel('recall')
            plt.ylabel('precision')
            ap_str = "{0:.2f}%".format(average_precision * 100)
            plt.title('Precision x Recall curve with confidence\nClass: %s, AP: %s' % (str(cls), ap_str))
            plt.legend(shadow=True)
            plt.grid()
            plt.savefig(os.path.join(self.savePath, cls + '.png'))
            if to_show:
                plt.show()
            plt.close()
            # plt.pause(0.05)

        """
         Finish counting true positives
        """
        for class_name in self.det_classes:
            # if class exists in detection-result but not in ground-truth then there are no true positives in that class
            if class_name not in self.gt_classes:
                tp_dict[class_name] = 0

        mAP = acc_AP / len(self.results)
        mAP_str = "{0:.2f}%".format(mAP * 100)
        print('mAP: %s over ground-truth %d classes' % (mAP_str, len(self.results)))

        # bar plot mAP and AP, save results:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP * 100)
        x_label = "Average Precision"
        output_path = self.savePath + "/mAP.png"
        to_show = False
        plot_color = 'royalblue'
        self.draw_plot_func(
            ap_dict,
            num_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )


        # detectin results true positive vs total detectin plot
        window_title = "detection-results-info"
        # Plot title
        plot_title = "detection-results\n"
        num_det_imgs = self.df_dr.imgID.nunique()
        plot_title += "(" + str(num_det_imgs) + " files and "
        num_det_classes= self.df_dr.class_name.nunique()
        plot_title += str(num_det_classes) + " detected classes)"
        # end Plot title
        x_label = "Number of objects per class"
        output_path = self.savePath + "/detection-results-info.png"
        to_show = False
        plot_color = 'forestgreen'
        self.draw_plot_func(
            self.num_det,
            num_det_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            tp_dict
        )

        return 'Evaluation Results Saved Successfully'

    def draw_plot_func(self, dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                       true_p_bar):
        # sort the dictionary by decreasing value, into a list of tuples
        sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
        # unpacking the list of tuples into two lists
        sorted_keys, sorted_values = zip(*sorted_dic_by_value)
        #
        if true_p_bar != "":  # this is for draw ground
            """
             Special case to draw in:
                - green -> TP: True Positives (object detected and matches ground-truth)
                - red -> FP: False Positives (object detected but does not match ground-truth)
                - pink -> FN: False Negatives (object not detected but present in the ground-truth)
            """
            fp_sorted = []
            tp_sorted = []
            for key in sorted_keys:
                fp_sorted.append(dictionary[key] - true_p_bar[key])
                tp_sorted.append(true_p_bar[key])
            plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
            plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                     left=fp_sorted)
            # add legend
            plt.legend(loc='lower right')
            """
             Write number on side of bar
            """
            fig = plt.gcf()  # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                fp_val = fp_sorted[i]
                tp_val = tp_sorted[i]
                fp_str_val = " " + str(fp_val)
                tp_str_val = fp_str_val + " " + str(tp_val)
                # trick to paint multicolor with offset:
                # first paint everything and then repaint the first number
                t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
                plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
                if i == (len(sorted_values) - 1):  # largest bar
                    self.adjust_axes(r, t, fig, axes)
        else:
            plt.barh(range(n_classes), sorted_values, color=plot_color)
            """
             Write number on side of bar
            """
            fig = plt.gcf()  # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                str_val = " " + str(val)  # add a space before
                if val < 1.0:
                    str_val = " {0:.2f}".format(val)
                t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
                # re-set axes to show number inside the figure
                if i == (len(sorted_values) - 1):  # largest bar
                    self.adjust_axes(r, t, fig, axes)
        # set window title
        fig.canvas.set_window_title(window_title)
        # write classes in y axis
        tick_font_size = 12
        plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
        """
         Re-scale height accordingly
        """
        init_height = fig.get_figheight()
        # comput the matrix height in points and inches
        dpi = fig.dpi
        height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
        height_in = height_pt / dpi
        # compute the required figure height
        top_margin = 0.15  # in percentage of the figure height
        bottom_margin = 0.05  # in percentage of the figure height
        figure_height = height_in / (1 - top_margin - bottom_margin)
        # set new height
        if figure_height > init_height:
            fig.set_figheight(figure_height)

        # set plot title
        plt.title(plot_title, fontsize=14)
        # set axis titles
        # plt.xlabel('classes')
        plt.xlabel(x_label, fontsize='large')
        # adjust size of window
        fig.tight_layout()
        # save the plot
        fig.savefig(output_path)
        # show image
        if to_show:
            plt.show()
        # close the plot
        plt.close()

    def adjust_axes(self, r, t, fig, axes):
        # get text width for re-scaling
        bb = t.get_window_extent(renderer=r)
        text_width_inches = bb.width / fig.dpi
        # get axis width in inches
        current_fig_width = fig.get_figwidth()
        new_fig_width = current_fig_width + text_width_inches
        propotion = new_fig_width / current_fig_width
        # get axis limit
        x_lim = axes.get_xlim()
        axes.set_xlim([x_lim[0], x_lim[1] * propotion])
