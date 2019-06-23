import numpy as np
import cv2
import time
import boto3

# --------- Connect to AWS --------
client = boto3.client('rekognition',
        region_name = 'us-east-1', 
        aws_access_key_id = 'aws_access_key_id',
        aws_secret_access_key = 'aws_secret_access_key')

# ---------- IP Cam URL ------
cap = cv2.VideoCapture(0)

while(True):
    start_time = time.time()
    
    ret, frame = cap.read()
    
    # Preprocess input
    height, width, channels = frame.shape
    jpeg_img = cv2.imencode('.jpg', frame)[1].tostring()
    
    # Run object detection
    response = client.detect_labels(Image={'Bytes':jpeg_img}, MinConfidence=80)
    if len(response['Labels']) > 0:  
        for label in response['Labels']:
            for instance in label['Instances']:
                label_name = label['Name']
                text_width, text_height = cv2.getTextSize(label_name,  cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=2)[0]
                x = int(instance['BoundingBox']['Left']*width)
                y = int(instance['BoundingBox']['Top']*height)
                w = int(instance['BoundingBox']['Width']*width)
                h = int(instance['BoundingBox']['Height']*height)

                # Display output
                cv2.rectangle(frame, (x,y), (x + text_width + 10, y + text_height + 20), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label_name, (x+5,y+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    else:
        cv2.putText(frame, 'NO OBJECT', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    
    # Dislpay FPS
    cv2.putText(frame, 'FPS: '+ str(1/(time.time()-start_time)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    # Dislpay Quit instruction
    cv2.putText(frame, 'Press Q to quit', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    # Display video stream
    cv2.imshow('frame',frame)
    
    # Reduce stress to processor
    time.sleep(0.1) 

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()