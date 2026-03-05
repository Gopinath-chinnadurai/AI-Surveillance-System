import os

print("AI Surveillance System")
print("1. Train Model")
print("2. Detect Image")
print("3. Webcam Detection")

choice = input("Enter your choice: ")

if choice == "1":
    os.system("python src/train_model.py")

elif choice == "2":
    os.system("python src/detect_image.py")

elif choice == "3":
    os.system("python src/webcam_detection.py")

else:
    print("Invalid choice")