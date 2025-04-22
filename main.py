import time

import cv2
import numpy as np
from paddleocr import PaddleOCR


# CPU上运行
class PlateReader(object):
    def __init__(self):
        self._img = None  #原始图像

        self._plateDetector = None  # yoloV5车牌检测模型（onnx）
        self._ocrModel = None  # 文本识别模型

        self._rects = None  # 非极大值抑制后的每个矩形的两点坐标
        self._conf = None  # 检测为车牌的置信度（绿色新能源小轿车车牌）
        self._plates = []  # 车牌图像
        self._platesText = []  # 每个车牌的文字
        self._keyCode = 0  # 按键反馈


        #加载车牌检测模型（yolov5）
        self._plateDetector = cv2.dnn.readNet(r'./yolov5_plate.onnx')  # yolov5新能源绿牌检测模型
        self._ocrModel = PaddleOCR(rec_algorithm='SVTR_LCNet', use_angle_cls=False)  # 导入paddle ocr轻量模型，关闭方向分类

        # 创建窗口
        cv2.namedWindow('detect result', cv2.WINDOW_NORMAL)

    # 处理数据输入格式为网络需要
    def yolov5_preprocess(self, img, target_size=640, transRB=True):
        """
        自定义YOLOv5预处理函数
        输入：BGR格式的OpenCV图像 (HWC)
        输出：符合ONNX模型要求的输入张量 (1,3,640,640) + 缩放比例/填充信息

        Returns:
            blob (np.ndarray): 预处理后的张量(1,3,H,W)
            ratio (float): 图像缩放比例
            padding (tuple): 上下或两侧的填充像素数 (top, left)
        """
        # 读取图像并保持BGR格式
        h, w = img.shape[:2]  # 原始高宽

        # 1. Letterbox缩放（保持长宽比填充灰边），计算缩小比例，以小的为标准
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 缩放图像
        resized = cv2.resize(img, (new_w, new_h))

        # 创建填充后的画布（114为YOLOv5默认填充值），默认大小640，640，3
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

        # 计算填充位置（居中）
        top = (target_size - new_h) // 2  # 存在画面的最端
        left = (target_size - new_w) // 2  # 存在画面的最左端
        padded[top:top + new_h, left:left + new_w] = resized  # 将resize过后的图像填充进去
        # 2. 转换数值范围与维度
        # BGR转RGB（若模型需要RGB输入则取消注释）
        if transRB:
            padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # 归一化到0~1并转float32
        normalized = padded.astype(np.float32) / 255.0

        # HWC -> CHW (3,640,640)
        chw = normalized.transpose(2, 0, 1)

        # 添加batch维度 (1,3,640,640)
        blob = np.expand_dims(chw, axis=0)

        # 返回blob，缩放比例，有图像的第一行、第一列、传给yolov5模型的图像
        return blob, scale, (top, left), padded

    # 还原原图中的坐标，得到（n，4）的矩形方框
    def getRect(self, yolo, scale, top, left):
        result = np.zeros((yolo.shape[0], 4), dtype=np.int32)
        # 先计算点
        result[:, 0] = yolo[:, 0] - yolo[:, 2] / 2
        result[:, 1] = yolo[:, 1] - yolo[:, 3] / 2
        result[:, 2] = yolo[:, 0] + yolo[:, 2] / 2
        result[:, 3] = yolo[:, 1] + yolo[:, 3] / 2
        # 计算缩放
        result[:, 0] = (result[:, 0] - left) / scale
        result[:, 1] = (result[:, 1] - top) / scale
        result[:, 2] = (result[:, 2] - left) / scale
        result[:, 3] = (result[:, 3] - top) / scale
        # 返回所有矩形的左上角点和右下角点
        return result

    #进行图像的车牌检测
    def detectPlate(self, rectThresh= 0.5, a1=0.4):
        # 预处理图像
        blob, scale, (top, left), padded = self.yolov5_preprocess(self._img, 640, True)
        # 处理网络输入
        self._plateDetector.setInput(blob)
        # 前向传播
        result = self._plateDetector.forward()[0]  #只有一张图像

        #将检测结果提取出，并将矩形转换为opencv两点格式
        result = result[np.where(result[:,4]>rectThresh)[0], :]  # 记得where返回的是一个元组
        resultRect = self.getRect(result, scale, top, left)

        # 进行非极大值抑制
        rectNMS = cv2.dnn.NMSBoxes(resultRect, result[:,4], rectThresh, a1)

        # 保存矩形两点以及置信度
        self._rects = resultRect[rectNMS, :]
        self._conf = result[rectNMS, 5:]

    # 针对提取的矩形区域，进行车牌提取和文本OCR识别
    def getText(self, textConf = 0.7, rectCorectLow = 2.7, rectCorectHigh = 3.5):
        # 文本清空
        self._platesText = []
        imgGray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        imgGray = cv2.equalizeHist(imgGray)
        # 处理每一个车牌
        if len(self._rects) == 0:
            print('None')
        for rect in self._rects:
            # print('车牌大小：', (rect[3]-rect[1], rect[2]-rect[0]))
            # plate = self._img[rect[1]:rect[3], rect[0]:rect[2]].copy()
            plate = imgGray[rect[1]:rect[3], rect[0]:rect[2]].copy()
            plate = cv2.equalizeHist(plate)
            plateH, plateW = plate.shape[:2]
            # 判断是否进行校正（简单拉伸校正）待检测四个点进行透视变换
            # print('车牌长宽比例：', plateW/plateH)
            if plateW/plateH>rectCorectHigh or plateW/plateH<rectCorectLow:
                # print('长宽校正')
                plate = cv2.resize(plate, None, None, plateH/plateW*3.14, 1)

            # cv2.imshow('plate', plate)

            # 车牌文本OCR识别
            plate = cv2.resize(plate, None, None, 2,2)  # 线性插值放大一下，ocr识别准确率提高（电脑摄像头比较小，导致车牌更小）
            textRects = self._ocrModel.ocr(plate, cls=False)  # 返回列表，多少个文本框，每个文本框多少行，每行四角坐标+文本与置信度

            #处理每一个文本框的每一个文本
            for textRect in textRects:
                linetext = ''
                if textRect is None:
                    print('空')
                    break
                # 每一个文本框
                for line in textRect:
                    # 每一行
                    if line[1][1]>textConf:
                        linetext = linetext+line[1][0]
                # 车牌正则化过滤
                pass

                # print('text: ', linetext)
                self._platesText.append(linetext[:2]+linetext[-6:])

    # 将车牌框打在图上
    def drawPlateRect(self):
        if self._img is not None:
            for rect in self._rects:
                cv2.rectangle(self._img, rect[:2], rect[2:], (0,0,255), 2)

    # 将车牌文字打图上
    def putPlateText(self):
        for i, text in enumerate(self._platesText):
            # 注意车牌有中文，无法正常显示
            cv2.putText(self._img, text, self._rects[i,:2], cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,255,255), 2)


    # 绘制信息并展示图片
    def showResult(self):
        self.drawPlateRect()
        self.putPlateText()
        cv2.imshow('detect result', self._img)
        self._keyCode = cv2.waitKey(0)

    # run
    def run(self, img, showResult = True, **params):
        # params: 参数：
        # rectThresh = 0.5  # 非极大值抑制时留下的矩形框的置信度
        # a1 = 0.4  # 非极大值抑制时抑制
        # textConf = 0.7  # 文本识别文本置信度
        # rectCorectLow = 2.7  # 车牌倾斜校正低阈值
        # rectCorectHigh = 3.5  # 车牌倾斜校正高阈值
        self._img = img

        self.detectPlate(rectThresh=params['rectThresh'], a1=params['a1'])

        self.getText(textConf=params['textConf'], rectCorectLow=params['rectCorectLow'], rectCorectHigh=params['rectCorectHigh'])

        if len(self._platesText)>0:
            print('text: ', self._platesText)

        if showResult:
            self.showResult()





if __name__ == '__main__':
    myPlateDetector = PlateReader()

    # ==================================图片模式
    img = cv2.imread(r'./imgs/plate1.jpeg')  # 注意图片不能太小，保证车牌（至少60行像素以上）
    print('img.shape: ', img.shape)

    img = cv2.GaussianBlur(img, (5,5), 0)

    myPlateDetector.run(img, True, rectThresh = 0.5, a1 = 0.4, textConf = 0.7,
                        rectCorectLow = 2.7, rectCorectHigh = 3.5)


    # ========================================摄像头、视频模式
    # camera = cv2.VideoCapture(0)
    # print('camera opened: ', camera.isOpened())
    #
    # ret, frame = camera.read()
    # while myPlateDetector._keyCode != 27 and ret:
    #     myPlateDetector.run(frame, True, rectThresh = 0.9, a1 = 0.4, textConf = 0.9,
    #                         rectCorectLow = 2.7, rectCorectHigh = 3.5)
    #     ret, frame = camera.read()
