from flask import Flask, render_template, url_for, Response
import cv2
import time
import dlib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    #Test the sending of frames with an image
    #filename = "me.jpg"
    #img = cv2.imread(filename)
    file_name = 'Hand_Detector.svm'
    detector = dlib.simple_object_detector(file_name)

    #Initialize the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    #Set the scale factor for the downsizing
    scale_factor = 3.0

    size, center_x = 0, 0

    fps = 0
    frame_counter = 0
    start_time = time.time()

    while cap.isOpened():

        #Read the camera frame by frame
        ret, img = cap.read()
        if not ret:
            break
        else:
            img = cv2.resize(img, (0,0), fx=2.0, fy=2.0)

            # Laterally flip the frame
            img = cv2.flip(img, 1 )
        
            # Calculate the Average FPS
            frame_counter += 1
            fps = (frame_counter / (time.time() - start_time))

            # Create a clean copy of the frame
            copy = img.copy() 

            # Downsize the frame.
            new_width = int(img.shape[1]/scale_factor)
            new_height = int(img.shape[0]/scale_factor)
            resized_frame = cv2.resize(copy, (new_width, new_height))

            # Detect with detector
            detections = detector(resized_frame)

            # Loop for each detection.
            for detection in (detections):    
                
                # Since we downscaled the image we will need to resacle the coordinates according to the original image.
                x1 = int(detection.left() * scale_factor )
                y1 = int(detection.top() * scale_factor )
                x2 = int(detection.right() * scale_factor )
                y2 = int(detection.bottom()* scale_factor )
                
                # Draw the bounding box
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0), 2 )
                cv2.putText(img, 'Hand Detected', (x1, y2+20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255),2)

                # Calculate size and center of the hand.
                size = int( (x2 - x1) * (y2-y1) )
                center_x = x2 - x1 // 2            

            # Display FPS and size of hand
            cv2.putText(img, 'FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255),2)

            # This information is useful for when you'll be building hand gesture applications
            cv2.putText(img, 'Center: {}'.format(center_x), (540, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
            cv2.putText(img, 'size: {}'.format(size), (540, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))            

            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')       

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)