from flask import Flask, render_template, Response
import cv2
from detect import detect_objects, labels

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        boxes, classes, scores = detect_objects(frame)

        imH, imW, _ = frame.shape
        for i in range(len(scores)):
            if scores[i] > 0.5:
                ymin = int(boxes[i][0] * imH)
                xmin = int(boxes[i][1] * imW)
                ymax = int(boxes[i][2] * imH)
                xmax = int(boxes[i][3] * imW)
                class_id = int(classes[i])
                label = labels[class_id]

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
