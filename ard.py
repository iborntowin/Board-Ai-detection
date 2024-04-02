import cv2
from ultralytics import YOLO


model = YOLO("../best8.pt")


target_classes = ["arduino-uno", "breadboard", "Capteur-IR-5 Voies", "capteur-ultrasonique",
                  "Clavier-Joystick", "ir-sensor", "module-bluetooth", "RF-module"]


#cap = cv2.VideoCapture("http://192.168.1.193:4747/video")
cap = cv2.VideoCapture(0)

class_counts = {class_name: 0 for class_name in target_classes}


color = (0, 0, 255)
offset = 30
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

while True:

    ret, img = cap.read()


    img = cv2.resize(img, (1000, 700))


    results = model.predict(source=img, conf=0.7)

    if results[0]:
        for x in results[0]:
            xyxy = x.boxes.xyxy.cpu().numpy().astype(int)
            class_id = x.boxes.cls.cpu().numpy().astype(int)[0]

            x1, y1 = xyxy[0][0], xyxy[0][1]
            x2, y2 = xyxy[0][2], xyxy[0][3]
            rectx1, recty1 = (x1 + x2) / 2, (y1 + y2) / 2
            rect_center = int(rectx1), int(recty1)
            cx, cy = rect_center

            if 0 <= class_id < len(model.names):
                class_name = model.names[class_id]

                if class_name in target_classes:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, class_name, (x1, y1), font, font_scale, (255, 255, 255), font_thickness)

                    if cx < (500 + offset) and cx > (500 - offset):

                        cv2.line(img, (500, 700), (500, 0), (0, 255, 0), 2)


                        count_text = f'{class_name} = {class_counts[class_name]}'
                        cv2.putText(img, count_text, (x1, y1 - 20), font, font_scale, (0, 0, 255), font_thickness)

        cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)




    cv2.imshow('Board Detection', img)

    # Check for key press
    k = cv2.waitKey(33)
    if k == 27:  
        break

cap.release()
cv2.destroyAllWindows()
