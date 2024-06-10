import os
import sys
import os
import cv2
import numpy as np
import argparse
import warnings
import time






from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
from utils import base64_to_pil




from PIL import Image







# Creating a new Flask Web application.
app = Flask(__name__)

print('Model loaded. Start serving...')


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
    result_text = "RealFace Score: {:.2f}".format(value) if label == 1 else "FakeFace Score: {:.2f}".format(value)
    color = (255, 0, 0) if label == 1 else (0, 0, 255)
    cv2.rectangle(frame, (image_bbox[0], image_bbox[1]), (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]), color, 2)
    cv2.putText(frame, result_text, (image_bbox[0], image_bbox[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)
    return frame

@app.route('/livevid')
def vidd():
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

def load_image_into_numpy_array(path):

    return np.array(Image.open(path))

def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return True
    else:
        return True


def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows();


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        
        # Get the image from post request
        img = base64_to_pil(request.json)

        #preds = test(img, modelpath, 0)
       
        
        return jsonify(result=preds)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_id", type=int, default=0, help="which gpu id, [0/1/2/3]")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models", help="model_lib used to test")
    parser.add_argument("--camera_id", type=int, default=0, help="ID of the camera device to use (default: 0)")
    args = parser.parse_args()
    http_server = WSGIServer(('0.0.0.0', 5003), app,)
    http_server.serve_forever()

