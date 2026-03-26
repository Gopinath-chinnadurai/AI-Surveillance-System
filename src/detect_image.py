from ultralytics import YOLO

model = YOLO(r"C:\AI_Surveillance_System\runs\detect\train\weights\best.pt")

image_path = r"C:\AI_Surveillance_System\dataset\combined_gunsnknifes\test\images\knife_83_jpg.rf.482135b05046c155e7fa630e25d8a72d.jpg"

results = model.predict(
    source=image_path,
    conf=0.4,
    save=True
)

print("Detection completed")