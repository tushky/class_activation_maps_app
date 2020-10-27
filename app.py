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
print(model)
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

@app.route('/cam/', methods=['POST', 'GET'])
def cam():

    if request.method == 'GET':
        return render_template('cam.html')
    
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    try:
        index = name_to_index[request.form['myCountry']]
        print('index:', index)
    except:
        index = None

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        image= Image.open(file).convert('RGB')
        image = read_image(image)
        activation_map.img = image
    elif not hasattr(activation_map, 'img'):
        return redirect(request.url)
    else:
        flash('Invalid File')
    cam, index = activation_map.show_cam(image, classifier, method='gradcam', class_index = index)
    cam = display_image(cam)
    flash('Image successfully uploaded and displayed')
    
    flash(f' Showing Class Actiovation Map For: {class_dict[int(index)]}')
    
    return render_template('cam.html', result=cam, named_class = class_names())

@app.route('/gradcam/', methods=['POST', 'GET'])
def gradcam():

    if request.method == 'GET':
        return render_template('gradcam.html')
    
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    try:
        index = name_to_index[request.form['myCountry']]
        print('index:', index)
    except:
        index = None
    print(f'selected label was : {index}')
    if file.filename == '' and not index:
        flash('No image or class selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        image=Image.open(file).convert('RGB')
        image = read_image(image)
        #index = None

    elif not hasattr(activation_map, 'img'):
        return redirect(request.url)
    cam, update_index = activation_map.show_cam(image, classifier, method='cam', class_index = index)
    result = display_image(cam)
    flash('Image successfully uploaded')
    if not index:
        index = update_index
    flash(f' Showing Class Actiovation Map For: {class_dict[int(index)]}')
    return render_template('gradcam.html', result=result, named_class=class_dict, selected=index)

@app.route('/gradcam++/', methods=['POST', 'GET'])
def gradcamplus():

    if request.method == 'GET':
        return render_template('gradcam++.html')
    
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    try:
        index = name_to_index[request.form['myCountry']]
        print('index:', index)
    except:
        index = None
    print(f'selected label was : {index}')
    if file.filename == '' and not index:
        flash('No image or class selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        image=Image.open(file).convert('RGB')
        image = read_image(image)

    elif not hasattr(activation_map, 'img'):
        return redirect(request.url)
    cam, update_index = activation_map.show_cam(image, classifier, method='gradcam++', class_index = index)
    result = display_image(cam)
    flash('Image successfully uploaded')
    if not index:
        index = update_index
    flash(f' Showing Class Actiovation Map For: {class_dict[int(index)]}')
    return render_template('gradcam++.html', result=result, named_class=class_dict, selected=index)

def display_image(image):
    #image = image.resize((640, 480))
    file_object = io.BytesIO()
    image.save(file_object, 'png')
    file_object.seek(0)
    base64img = "data:image/png;base64," + b64encode(file_object.getvalue()).decode('ascii')
    return base64img

if __name__ == "__main__":
    app.run(debug=True)