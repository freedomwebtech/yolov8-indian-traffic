import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

from vidgear.gears import CamGear

model=YOLO('best.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
            
        



cap=cv2.VideoCapture('indianroadtraffic.mp4')


my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

cy1=426
offset=6

count=0

while True:    
    ret,frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
   

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
   

#    print(px)
    list=[]
    list1=[]
    list2=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        if cy1<(cy+offset) and cy1>(cy-offset):
           cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
           cvzone.putTextRect(frame,f'{c}',(x2,y2),1,1)
        
            
    cv2.line(frame,(355,426),(639,426),(255,255,255),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
#cap.release()
stream.stop()    
cv2.destroyAllWindows()




