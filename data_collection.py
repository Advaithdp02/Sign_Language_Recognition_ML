import os
import cv2

data_dir='./Data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


num_of_classes=31
dataset_size=200

cap=cv2.VideoCapture(1) #Opens a connection to the default camera(0)

for j in range(num_of_classes):
    if not os.path.exists(os.path.join(data_dir, str(j))):
        os.makedirs(os.path.join(data_dir, str(j))) #This sets up a unique folder for each class of images (e.g., ./Data/0, ./Data/1, ./Data/2).

    print('Collecting data for class {}'.format(j))

    done=False
    while True:
        ret, frame = cap.read() # Captures a frame from the camera feed.
        cv2.putText(frame, f'Ready? Press "Q" ! :) {j}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA) #Adds a text message to the frame, instructing the user to press "Q" when ready.
        cv2.imshow('frame', frame) #: Displays the frame with the text.
        if cv2.waitKey(25) == ord('q'):
            break


    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read() # Captures a frame from the camera.
        cv2.imshow('frame', frame)
        cv2.waitKey(25) # Adds a slight delay between frames.
        cv2.imwrite(os.path.join(data_dir, str(j), '{}.jpg'.format(counter)), frame) # Saves each captured frame as a .jpg file in the corresponding class folder (e.g., ./Data/0/0.jpg, ./Data/0/1.jpg).

        counter += 1
cap.release() #Releases the camera resource.
cv2.destroyAllWindows() #Closes all OpenCV windows.