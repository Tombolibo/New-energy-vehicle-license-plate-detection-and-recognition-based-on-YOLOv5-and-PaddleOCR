import os
import time

import cv2
import numpy as np
from PIL import Image

#获取图像高宽
def getImgSize(filePath):
    with Image.open(filePath) as img:
        return img.size

def getPlatePos(filePath):
    fileName = os.path.basename(filePath).split('.')[0]
    width, height = getImgSize(filePath)
    # 通过文件名获取车牌信息
    # 车牌面积占比、水平+垂直倾斜角度、标注框坐标（矩形框左上右下坐标+车牌四个点右下、左下、左上、右上顺序）、车牌字符映射、亮度、模糊度
    area, tilt, box, points, label, brightness, blurriness = fileName.split('-')
    platePoints = np.array([np.array(pt.split('&'), dtype=np.int32) for pt in points.split('_')], dtype=np.int32)
    return platePoints

def getRectPos(filePath, classID = 0, savedir = None):
    fileName = os.path.basename(filePath).split('.')[0]  # 去掉后面的后缀名.jpg等
    width, height = getImgSize(filePath)
    # 通过文件名获取车牌信息
    # 车牌面积占比、水平+垂直倾斜角度、标注框坐标（矩形框左上右下坐标+车牌四个点右下、左下、左上、右上顺序）、车牌字符映射、亮度、模糊度
    area, tilt, box, points, label, brightness, blurriness = fileName.split('-')
    box_str = box.replace('&',' ').replace('_',' ')
    rectPoints = np.fromstring(box_str, dtype=np.int32, sep = ' ').reshape(2,2)
    #计算yolov5需要格式：中心x、中心y、width、height（都是原图的百分比形式）
    yoloPos = np.array([classID,
                        (rectPoints[0,0]+rectPoints[1,0])//2/width,
                       (rectPoints[0,1]+rectPoints[1,1])//2/height,
                       (rectPoints[1,0]-rectPoints[0,0])/width,
                       (rectPoints[1,1]-rectPoints[0,1])/height],
                       dtype=np.float32)
    if savedir is not None:
        np.savetxt(os.path.join(savedir,fileName+'.txt'), [yoloPos], fmt='%d %.6f %.6f %.6f %.6f')
    return yoloPos

# 处理图片文件夹，返回yolov5训练集格式
def getDataLabel(datasetPath,classID=0, savedir = None):
    fileList = os.listdir(datasetPath)
    for filename in fileList:
        getRectPos(os.path.join(datasetPath, filename), classID, savedir)



if __name__ == '__main__':

    time1 = time.time()

    getDataLabel(r'C:\Users\11971\OneDrive\Python_project\data\CCPD2020\CCPD2020\ccpd_green\val\images',
                 0,
                 r'C:\Users\11971\OneDrive\Python_project\data\CCPD2020\CCPD2020\ccpd_green\val\labels')

    time2 = time.time()

    print((time2-time1)*1000, 'ms')
