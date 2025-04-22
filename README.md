# New-energy-vehicle-license-plate-detection-and-recognition-based-on-YOLOv5-and-PaddleOCR
## 基于yolov5和paddleocr的新能源小汽车车牌检测定位和文本识别

支持对图片、视频、摄像头进行车牌检测和文本识别并实时标注。<br/>

## yolov5车牌检测
使用中国新能源汽车数据集CCPD2020，对yolov5模型进行训练（仅一类，新能源小汽车绿牌），模型指标：<br/>
train/box_loss: 0.0092943<br/>
train/obj_loss: 0.0028724<br/>
train/cls_loss: 0<br/>
metrics/precision: 0.99582<br/>
metrics/recall: 0.95158<br/>
metrics/mAP_0.5: 0.97959<br/>
metrics/mAP_0.5:0.95: 0.80126<br/>
val/box_loss: 0.01654<br/>
val/obj_loss: 0.0049936<br/>
val/cls_loss: 0<br/>
x/lr0: 0.000496<br/>
x/lr1: 0.000496<br/>
x/lr2: 0.000496<br/>
训练完成后将模型转换onnx导入到opencvdnn框架中使用。<br/>

## 车牌文本识别
进行车牌的大致倾斜校正，使用PaddleOCR对所定位车牌进行文本识别，设置阈值较高，并于图像上标注实时识别结果。


