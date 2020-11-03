'''
    Created By: Tushar Gadhiya
'''
import os
import io
from base64 import b64encode
from flask import Flask, flash, request, redirect, render_template
from PIL import Image

from torchvision import models
from class_activation_maps import ClassActivationMaps, SubNet
from utils import read_image
from classes import class_names


model = models.mobilenet_v2(pretrained=True)
activation_map = ClassActivationMaps(model)
classifier = SubNet(model)
UPLOAD_FOLDER = 'static/images/'
class_dict, name_to_index = class_names()
app = Flask(__name__)
app.secret_key = "secret key"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'JPEG'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/<method>', methods=['POST', 'GET'])
def get_cam(method):
    fency_name = ''
    if method == 'cam' : fency_name = 'CAM: Class Activation Map'
    elif method == 'gradcam' : fency_name = 'Grad-CAM: Gradient Weighted Class Activation Map'
    elif method == 'gradcam++' : fency_name = 'Grad-CAM++: Gradient Weighted Class Activation Map++'
    if request.method == 'GET':
        return render_template('get_cam.html', result=False, method=method, fency_name=fency_name)
    
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    try:
        index = name_to_index[request.form['class_name']]
        print('index:', index)
    except:
        index = None

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        try:
            image= Image.open(file).convert('RGB')
            image = read_image(image)
        except:
            flash('Invalid file')
            return redirect(request.url)
    else:
        flash('Invalid file')
        return redirect(request.url)
    cam, index = activation_map.show_cam(image, classifier, method=method, class_index = index)
    cam = display_image(cam)
    flash('Image successfully uploaded and displayed')
    
    flash(f' Showing Class Actiovation Map For: {class_dict[int(index)]}')
    
    return render_template('get_cam.html', result=cam, named_class = class_names(), method=method, fency_name=fency_name)

def display_image(image):
    #image = image.resize((640, 480))
    file_object = io.BytesIO()
    image.save(file_object, 'png')
    file_object.seek(0)
    base64img = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')
    return base64img

if __name__ == "__main__":
    app.run()