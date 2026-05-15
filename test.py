import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    cv2.imshow("cam", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    print("testing")