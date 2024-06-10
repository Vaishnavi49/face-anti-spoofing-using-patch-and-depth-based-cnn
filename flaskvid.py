from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
import os
import argparse
import warnings

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

app = Flask(__name__)

class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.model_test = AntiSpoofPredict(0)
        self.image_cropper = CropImage()

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = test_frame(frame, self.model_test, self.image_cropper)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

video_camera = None

def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\n Height/Width should be 4/3.")
        return True
    else:
        return True

def test_frame(frame, model_test, image_cropper):
    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    for model_name in os.listdir(args.model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(args.model_dir, model_name))
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    result_text = "RealFace Score: {:.2f}".format(value) if label == 1 else "Spoofface Score: {:.2f}".format(value)
    color = (255, 0, 0) if label == 1 else (0, 0, 255)
    cv2.rectangle(frame, (image_bbox[0], image_bbox[1]), (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]), color, 2)
    cv2.putText(frame, result_text, (image_bbox[0], image_bbox[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.9 * frame.shape[0] / 1024, color)
    return frame

@app.route('/')
def index():
    return render_template('vidd.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()
    return Response(gen(video_camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global video_camera
    if video_camera is None:
        video_camera = VideoCamera()
    return 'Camera started'

@app.route('/stop_camera')
def stop_camera():
    global video_camera
    if video_camera is not None:
        del video_camera
        video_camera = None
    return 'Camera stopped'

if __name__ == '__main__':
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models", help="model_lib used to test")
    parser.add_argument("--camera_id", type=int, default=0, help="ID of the camera device to use (default: 0)")
    args = parser.parse_args()
    app.run(debug=True)

