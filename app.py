import glob
import io
import re

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from flask import Flask, jsonify, render_template, request
from flask.scaffold import F
from monai.networks.nets.densenet import Densenet121
from PIL import Image
from torch import nn
from torch._C import device
from torch.utils import data

SIZE = 256

def build_model():
    model = Densenet121(spatial_dims=3,
                        in_channels=1, out_channels=1)
    return model

model = build_model()
checkpoint = torch.load("T1wCE-e18-loss0.727-auc0.650.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'])
# model.load("./T1wCE-e18-loss0.727-auc0.650.pth", device="cpu")
model.eval()

def load_dicom_image(path, img_size=SIZE):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if np.min(data) == np.max(data):
        data =  np.zeros((256, 256), dtype=float)
        return data
    data = cv2.resize(data, (img_size, img_size))
    return data

files = ["Image-1.dcm", "Image-10.dcm"]

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def load_dicom_images_3d(img_size=256, mri_type="FLAIR"):    
    files_to_load = natural_sort(glob.glob("./data/*.dcm"))
    img3d = np.stack([load_dicom_image(path=f, img_size=img_size) for f in files_to_load]).T    
    img3d = img3d - np.min(img3d)
    if np.max(img3d) != 0:
        img3d = img3d / np.max(img3d)
    
    return np.expand_dims(np.expand_dims(img3d, 0), 0)

print(len(glob.glob("./data/*.dcm")))
# print(load_dicom_images_3d("./data/*.dcm", mri_type="FLAIR").shape)
    

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return jsonify({"msg": "try posting to the /predict endpoint with a dicom file"})

@app.route('/upload')
def upload_file():
   return render_template('./index.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # if request.method == 'POST':
        # file = request.files['file']
    # if file is not None:
    input_tensor = load_dicom_images_3d(img_size=256, )
    with torch.no_grad():
        output = torch.sigmoid(model(torch.tensor(input_tensor).float()).squeeze(1)).numpy().squeeze().tolist()
    # output = pd.Series(output).to_json(orient='split')
    return jsonify({"Prediction": output})
    # return jsonify({"Prediction": "nothing"})

if __name__=='__main__':
    app.run(debug=True)
    
    

