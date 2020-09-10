import os
from app import app
import urllib.request
from base64 import b64encode
from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from PIL import Image
from model import DeconvNet
from grad_cam import GradCAM
from utils import load_image
from classes import class_names
import io
from saliency import SaliencyMap
from torchvision import models
from cam import CAM
from utils import tensor_to_image

googlenet = models.googlenet(pretrained=True)
resnet34 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)

active_models = {}

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def reset_model():
    return DeconvNet(5, True)
	
@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/cam/', methods=['POST', 'GET'])
def cam():

    if request.method == 'GET':
        active_models['cam'] = CAM(googlenet)
        print('saving cam model')
        return render_template('cam.html')
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    
    if active_models.get('cam'):
            print('exsiting model loaded')
            cam = active_models['cam']
    else:
        cam = GradCAM(googlenet)

    file = request.files['file']
    index = None if 'select_class' not in request.form else request.form['select_class']

    if file.filename == '' and not index:
        flash('No image or class selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        image=Image.open(file).convert('RGB')
        cam.img = load_image(image)
        index = None

    elif not hasattr(cam, 'img'):
        return redirect(request.url)
    print('index is : ',index)
    out = cam.get_cam(int(index) if index else None)
    print(out.size)
    out = display_image(out)
    flash('Image successfully uploaded and displayed')
    class_dict = class_names()
    if not index:
        index = cam.cnn(cam.img).argmax().item()
    flash(class_dict[int(index)])
    #print(filename)
    return render_template('cam.html', filename=out, named_class = class_names())

@app.route('/gradcam/', methods=['POST', 'GET'])
def gradcamm():

    if request.method == 'GET':
        active_models['gradcam'] = GradCAM(googlenet)
        print('saving gradcam model')
        return render_template('gradcam.html')
    
    if 'file' not in request.files:
        return redirect(request.url)

    
    if active_models.get('gradcam'):
            print('exsiting model loaded')
            gradcam = active_models['gradcam']
    else:
        gradcam = GradCAM(googlenet)

    file = request.files['file']
    index = None if 'select_class' not in request.form else request.form['select_class']

    if file.filename == '' and not index:
        flash('No image or class selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        image=Image.open(file).convert('RGB')
        gradcam.img = load_image(image)
        index = None

    elif not hasattr(gradcam, 'img'):
        return redirect(request.url)
    print('index is : ',index)
    cam = gradcam.show_cam(int(index) if index else None)
    print(cam.size)
    original = display_image(tensor_to_image(gradcam.img))
    result = display_image(cam)
    flash('Image successfully uploaded and displayed')
    class_dict = class_names()
    if not index:
        index = gradcam.cnn(gradcam.img).argmax().item()
    flash(class_dict[int(index)])
    #print(filename)
    return render_template('gradcam.html', original=original, result=result, named_class = class_names())

@app.route('/saliency/', methods=['POST', 'GET'])
def saliency():
    if request.method == 'GET':
        active_models['saliency'] = SaliencyMap(googlenet)
        return render_template('saliency.html')
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    
    file = request.files['file']

    if file.filename == '' and index=='':
        flash('No image or class selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        saliency = SaliencyMap(googlenet)
        image=Image.open(file).convert('RGB')
        img = load_image(image)
    else:
        return redirect(request.url)

    if active_models.get('saliency'):
        saliency = active_models['saliency']
    else:
        saliency = SaliencyMap(googlenet)
    out = saliency(img)
    print(out.size)
    out = display_image(out)
    flash('Image successfully uploaded and displayed')
    return render_template('saliency.html', filename=out)


def display_image(image):
    #image = image.resize((640, 480))
    file_object = io.BytesIO()
    image.save(file_object, 'PNG')
    file_object.seek(0)
    base64img = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')
    return base64img

if __name__ == "__main__":
    app.run(debug=True)