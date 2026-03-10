"""
Find all available cameras and test them
Run this to see which camera number is DroidCam
"""

import cv2

print("Testing available cameras...\n")

for i in range(10):  # Test camera indices 0-9
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i}: WORKING")
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
            
            # Show preview for 2 seconds
            cv2.imshow(f"Camera {i} - Press any key", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"Camera {i}: Not available")

print("\n" + "="*50)
print("Which camera showed your phone's view?")
print("="*50)
