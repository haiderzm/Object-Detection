from flask import Flask, render_template, request
from PIL import Image
import base64
import io
import cv2
import numpy as np
import torch

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
model= torch.load('obj_det.pt')
model.eval()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/out', methods=['POST'])
def out():
    file = request.files['imagefile'].read() ## byte file
    npimg = np.fromstring(file, np.uint8)
    # print(type(npimg))
    img = cv2.cvtColor(cv2.imdecode(npimg,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    # print(type(img))
	
    res = model(img)
    img_n = np.squeeze(res.render())

    img_n = Image.fromarray(img_n.astype("uint8"))
    # print(type(img_n))
    rawBytes = io.BytesIO()
    img_n.save(rawBytes, "JPEG")
    # print(type(img_n))
    
    img_base64 = base64.b64encode(rawBytes.getvalue())
    # print(type(img_base64))
    
    return render_template('out.html', img_data=img_base64.decode('utf-8'))

if __name__ == '__main__':
    app.run(debug=True)