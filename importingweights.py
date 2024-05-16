import torch.hub

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='local')
