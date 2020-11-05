import os
import matplotlib.pyplot as plt
from Evaluator import *
import pdb
import pandas as pd


# tt 20200721, I wrote this txt to csv to test my csv algoritim
def txt_to_csv(Folder, split='GT'):
    files = os.listdir(Folder)
    files.sort()
    row = []

    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(os.path.join(Folder, f), "r")

        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if split == 'GT':
                cls = (splitLine[0])  # class
                left = float(splitLine[1])
                top = float(splitLine[2])
                right = float(splitLine[3])
                bottom = float(splitLine[4])
                row.append({'imgID': nameOfImage, 'width': -1, 'height': -1, 'class_name': cls, 'xmin': left,
                            'ymin': top, 'xmax': right, 'ymax': bottom})
            else:  # 'detection-results'
                cls = splitLine[0]
                left = float(splitLine[2])
                top = float(splitLine[3])
                right = float(splitLine[4])
                bottom = float(splitLine[5])
                score = float(splitLine[1])
                row.append({'imgID': nameOfImage, 'class_name': cls, 'confidence': score,
                            'xmin': left, 'ymin': top, 'xmax': right, 'ymax': bottom})
        fh1.close()
    if split == 'GT':
        df_data = pd.DataFrame(row, columns=['imgID', 'width', 'height', 'class_name', 'xmin', 'ymin', 'xmax', 'ymax'])
    else:
        df_data = pd.DataFrame(row, columns=['imgID', 'class_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
    df_data.to_csv(Folder + '.csv', index=False)
    return df_data


if __name__ == '__main__':
    GTFolder = 'C:\\Users\\tangjny\\Station\\work_station\\Software\\Tools\\model-evaluation\\data\\ground-truth'
    DetFolder = 'C:\\Users\\tangjny\\Station\\work_station\\Software\\Tools\\model-evaluation\\data\\detection-results'
    df_gt = txt_to_csv(GTFolder, split='GT')
    df_det = txt_to_csv(DetFolder, split='Det')
    print('completed conver txt to csv!')