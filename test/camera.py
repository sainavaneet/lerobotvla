import cv2

def main():
    # Initialize the camera (0 is usually the default webcam)
    cap = cv2.VideoCapture(6)
    
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
        cv2.imwrite('Camera Feed.png', frame)
        
  
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
