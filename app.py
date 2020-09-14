import os
from base64 import b64encode
from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from PIL import Image
from grad_cam import GradCAM, SubNet
from utils import load_image
from classes import class_names
#from saliency import SaliencyMap
from torchvision import models
#from cam import CAM
from utils import tensor_to_image
import io

googlenet = models.mobilenet_v2(pretrained=True)
#print(googlenet)
gradcam = GradCAM(googlenet, name='features_18')
model = SubNet(googlenet)
UPLOAD_FOLDER = 'static/images/'

app = Flask(__name__)
app.secret_key = "secret key"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/cam/', methods=['POST', 'GET'])
def cam():

    if request.method == 'GET':
        return render_template('cam.html')
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No image or selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        image= Image.open(file).convert('RGB')
        image = load_image(image)

    elif not hasattr(cam, 'img'):
        return redirect(request.url)
    out, index = gradcam.get_cam(image)
    print(out.size)
    out = display_image(out)
    flash('Image successfully uploaded and displayed')
    class_dict = class_names()
    flash(f"Predicted class : {class_dict[int(index)]}")
    
    return render_template('cam.html', filename=out, named_class = class_names())

@app.route('/gradcam/', methods=['POST', 'GET'])
def gradcamm():

    if request.method == 'GET':
        return render_template('gradcam.html')
    
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    try:
        index = int(request.form['label'])
    except:
        index = None
    print(f'selected label was : {index}')
    if file.filename == '' and not index:
        flash('No image or class selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        image=Image.open(file).convert('RGB')
        image = load_image(image)
        #index = None

    elif not hasattr(gradcam, 'img'):
        return redirect(request.url)
    cam, update_index = gradcam.get_gradcam(image, model, index)
    result = display_image(cam)
    flash('Image successfully uploaded')
    class_dict = class_names()
    if not index:
        index = update_index
    flash(f' Predicted class: {class_dict[int(index)]}')
    return render_template('gradcam.html', result=result, named_class=class_dict, selected=index)
"""
@app.route('/saliency/', methods=['POST', 'GET'])
def saliency():
    if request.method == 'GET':
        return render_template('saliency.html')
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    
    file = request.files['file']

    if file.filename == '':
        flash('No image or class selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        image=Image.open(file).convert('RGB')
        img = load_image(image)
    else:
        return redirect(request.url)

    saliency = SaliencyMap(googlenet)

    out = saliency(img)
    print(out.size)
    out = display_image(out)
    flash('Image successfully uploaded')
    return render_template('saliency.html', filename=out)
"""

def display_image(image):
    #image = image.resize((640, 480))
    file_object = io.BytesIO()
    image.save(file_object, 'png')
    file_object.seek(0)
    base64img = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')
    return base64img

if __name__ == "__main__":
    app.run()