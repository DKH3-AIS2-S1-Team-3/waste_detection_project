from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")  # update path if needed

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Run detection
    results = model.predict(source=frame, conf=0.25, verbose=False)

    # Display predictions on frame
    annotated_frame = results[0].plot()  # this adds bounding boxes
    cv2.imshow("Waste Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()