# Import Libraries
import dlib
import cv2
import os
import time
import numpy as np
import shutil

def main():
    # If cleanup is True then the new images and annotations will be appended to previous ones
    cleanup = True

    #Setup the window and initialize webcam
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1920,1080)
    cv2.moveWindow("frame", 0,0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Initalize sliding window's x1,y1 
    x1 ,y1 = 0,0
    window_width = 190
    window_height = 190

    # We will save images after every 4 frames to prevent duplicates
    skip_frames = 4
    frame_gap = 0

    # Store images in rgis directory
    directory = 'train_images_h'
    box_file = 'boxes_h.txt'

    if cleanup:
        
        # Delete the images directory if it exists
        if os.path.exists(directory):
            shutil.rmtree(directory)
        
        # Clear up all previous bounding boxes
        open(box_file, 'w').close()
        
        # Initialize the counter to 0
        counter = 0
        
    elif os.path.exists(box_file):
        #Append new boxes to the previously stored ones
        with open(box_file,'r') as text_file:
            box_content = text_file.read()
            
        # Set the counter to the previous highest checkpoint
        counter = int(box_content.split(':')[-2].split(',')[-1])

    # Open up this text file or create it if it does not exists
    fr = open(box_file, 'a')

    # Create our image directory if it does not exists.
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Initial wait before you start recording each row
    initial_wait = 0
            
    while(True):
        
        # Start reading from camera
        ret, frame = cap.read()
        if not ret:
            break
            
        # Invert the image laterally to get the mirror reflection.
        frame = cv2.flip( frame, 1 )
        
        # Make a copy of the original frame
        orig = frame.copy()    
        
        # Wait the first 50 frames so that you can place your hand correctly
        if initial_wait > 60:

            frame_gap +=1  
        
            # Move the window to the right 
            if x1 + window_width < frame.shape[1]:
                x1 += 4
                time.sleep(0.1)            
                
            elif y1 + window_height + 270 < frame.shape[1]:

                # If the sliding_window has reached the end of the row then move down
                y1 += 80    
                x1 = 0
                frame_gap = 0
                initial_wait = 0
                
            # Break the loop if we have gone over the whole screen.
            else:
                break
                
        else: 
            initial_wait += 1

        # Save the image every nth frame.
        if frame_gap == skip_frames:

            # Set the image name equal to the counter value
            img_name = str(counter)  + '.png'

            # Save the Image in the defined directory
            img_full_name = directory + '/' + str(counter) +  '.png'
            cv2.imwrite(img_full_name, orig)
            
            # Save the bounding box coordinates in the text file.
            fr.write('{}:({},{},{},{}),'.format(counter, x1, y1, x1+window_width, y1+window_height))

            # Increment the counter 
            counter += 1

            # Set the frame_gap back to 0.
            frame_gap = 0

        # Draw the sliding window
        cv2.rectangle(frame,(x1,y1),(x1+window_width,y1+window_height),(0,255,0),3)
        
        # Display the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
                    break

    # Release camera and close the file and window
    cap.release()
    cv2.destroyAllWindows()
    fr.close()
    print("[INFO] Data collection complete...")

    # In this dictionary our images and annotations will be stored.
    data = {}

    # Get the indexes of all images.
    image_indexes = [int(img_name.split('.')[0]) for img_name in os.listdir(directory)]

    # Shuffle the indexes to have random train/test split later on.
    np.random.shuffle(image_indexes)

    # Open and read the content of the boxes.txt file
    f = open(box_file, "r")
    box_content = f.read()

    # Convert the bounding boxes to dictionary in the format `index: (x1,y1,x2,y2)` ...
    box_dict =  eval( '{' +box_content + '}' )

    # Close the file
    f.close()

    # Loop over all indexes
    for index in image_indexes:
        
        # Read the image in memmory and append it to the list
        img = cv2.imread(os.path.join(directory, str(index) + '.png'))    
        
        # Read the associated bounding_box
        bounding_box = box_dict[index]
        
        # Convert the bounding box to dlib format
        x1, y1, x2, y2  = bounding_box
        dlib_box = [ dlib.rectangle(left=x1 , top=y1, right=x2, bottom=y2) ]
        
        # Store the image and the box together
        data[index] = (img, dlib_box)

    # This is the percentage of data we will use to train
    # The rest will be used for testing
    percent = 0.8

    # How many examples make 80%.
    split = int(len(data) * percent)

    # Seperate the images and bounding boxes in different lists.
    images = [tuple_value[0] for tuple_value in data.values()]
    bounding_boxes = [tuple_value[1] for tuple_value in data.values()]

    # Initialize object detector Options
    options = dlib.simple_object_detector_training_options()

    # I'm disabling the horizontal flipping, becauase it confuses the detector 
    # if you're training on few examples
    options.add_left_right_image_flips = False

    # Set the C parameter of SVM 
    options.C = 5

    # Note the start time before training.
    st = time.time()

    # You can start the training now
    print("[INFO]Beginning training of the model...")
    detector = dlib.train_simple_object_detector(images[:split], bounding_boxes[:split], options)

    # Print the Total time taken to train the detector
    print('Training Completed, Total Time taken: {:.2f} seconds'.format(time.time() - st))

    file_name = 'Hand_Detector.svm'
    detector.save(file_name)

    win_det = dlib.image_window()
    win_det.set_image(detector)

    print("\nTraining Metrics: {}".format(dlib.test_simple_object_detector(images[:split], bounding_boxes[:split], detector)))

    print("Testing Metrics: {}".format(dlib.test_simple_object_detector(images[split:], bounding_boxes[split:], detector)))

if __name__ == "__main__":
    main()
