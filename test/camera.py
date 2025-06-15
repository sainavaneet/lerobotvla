import cv2

def check_available_cameras():
    available_cameras = []
    for i in range(15):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def main():
    # Check available cameras
    available_cameras = check_available_cameras()
    if not available_cameras:
        print("Error: No cameras found")
        return
    
    print(f"Available cameras: {available_cameras}")
    

    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame")
            break
            
        # Display the frame
        # frame = cv2.resize(frame, (480, 640))
        cv2.imwrite('Camera Feed.png', frame)
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
