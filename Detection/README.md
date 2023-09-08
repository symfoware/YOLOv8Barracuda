# YOLOv8Barracuda
YOLOv8 Detection + Unity Barracuda sample
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
import onnx

def fix_onnx_resize(model):
    for i in range(len(model.graph.node)):
        node = model.graph.node[i]
        if node.op_type != 'Resize':
            continue
        
        new_node = onnx.helper.make_node(
                'Resize',
                inputs=node.input,
                outputs=node.output,
                name=node.name,
                coordinate_transformation_mode='half_pixel',  # Instead of pytorch_half_pixel, unsupported by Tensorflow
                mode='linear',
            )
        model.graph.node.insert(i, new_node)
        model.graph.node.remove(node)


model = YOLO('yolov8n.pt')
model.export(format='onnx', opset=12, simplify=True)
onnx_model = onnx.load('yolov8n.onnx')

fix_onnx_resize(onnx_model)
onnx.save(onnx_model, 'yolov8n-barracuda.onnx')
```


