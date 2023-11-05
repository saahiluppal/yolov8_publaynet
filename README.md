yolo detect train data=data.yaml model=yolov8m.yaml pretrained=yolov8m.pt epochs=100 batch=36 imgsz=640 name=yolov8m_custom device=0,1,2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
