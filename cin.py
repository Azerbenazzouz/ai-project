from ultralytics import YOLO

model2 = YOLO('cin.pt')
model2.predict(source=0, imgsz=640,conf=0.6, show= True)

