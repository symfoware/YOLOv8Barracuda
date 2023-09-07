# YOLOv8Barracuda
YOLOv8 Classification + Unity Barracuda sample
* Unity: 2022.3.8f1
* Barracuda: 3.0.0

## Pythonによるモデルの変換方法
YOLOv8モデルを取得  
https://github.com/ultralytics/ultralytics

ultralyticsをインストール
```
pip install ultralytics
```

以下のコードでモデルを変換

```python:convert.py
from ultralytics import YOLO
model = YOLO('yolov8n-cls.pt')
model.export(format='onnx', opset=12)
```


