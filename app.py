from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Class names from your dataset - updated with your specific classes
class_names = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        processed_image = preprocess_image(file_path)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Get disease information
        disease_info = get_disease_info(predicted_class)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'prediction': predicted_class,
            'confidence': confidence,
            'disease_info': disease_info
        })
    
    return jsonify({'error': 'File type not allowed'})

def get_disease_info(disease_class):
    # Dictionary containing information about each disease - updated for your specific classes
    disease_info = {
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
            'symptoms': 'Long, narrow, rectangular lesions that run parallel to leaf veins. Initially pale brown, becoming gray with dark borders as they mature.',
            'treatment': 'Apply appropriate fungicides. Rotate crops and remove infected debris.',
            'prevention': 'Plant resistant hybrids, practice crop rotation, and ensure adequate spacing for air circulation.'
        },
        'Corn_(maize)___Common_rust_': {
            'symptoms': 'Small, round to elongated pustules on both leaf surfaces. Pustules are reddish-brown and develop in circular or elongated clusters.',
            'treatment': 'Apply fungicides in early stages. Remove heavily infected plants.',
            'prevention': 'Plant resistant hybrids and ensure proper field aeration and drainage.'
        },
        'Corn_(maize)___Northern_Leaf_Blight': {
            'symptoms': 'Long, elliptical lesions on leaves that are grayish-green to tan. Lesions may develop dark areas as they produce spores.',
            'treatment': 'Apply fungicides when disease first appears. Remove and destroy infected plants.',
            'prevention': 'Plant resistant hybrids, rotate crops, and practice good field sanitation.'
        },
        'Corn_(maize)___healthy': {
            'symptoms': 'No symptoms of disease. Leaves show normal color and structure.',
            'treatment': 'No treatment needed.',
            'prevention': 'Maintain good agricultural practices including proper spacing, adequate fertilization, and irrigation.'
        },
        'Potato___Early_blight': {
            'symptoms': 'Dark brown to black lesions with concentric rings forming a "target spot" pattern. Usually begins on lower leaves.',
            'treatment': 'Apply fungicides preventatively. Remove and destroy infected leaves.',
            'prevention': 'Rotate crops, space plants for good air circulation, and avoid overhead irrigation.'
        },
        'Potato___Late_blight': {
            'symptoms': 'Pale to dark green water-soaked spots on leaves, eventually turning brown to purplish-black. White fuzzy growth may appear on underside of leaves in humid conditions.',
            'treatment': 'Apply copper-based fungicides immediately. Remove infected plants entirely to prevent spread.',
            'prevention': 'Plant resistant varieties, destroy volunteer potato plants, and practice crop rotation.'
        },
        'Potato___healthy': {
            'symptoms': 'No symptoms of disease. Leaves show normal color and structure.',
            'treatment': 'No treatment needed.',
            'prevention': 'Maintain proper cultural practices including crop rotation and adequate spacing.'
        },
        'Tomato___Bacterial_spot': {
            'symptoms': 'Small, dark, water-soaked, circular spots on leaves, stems, and fruit. Spots may have a yellow halo and become angular as they enlarge.',
            'treatment': 'Apply copper-based bactericides. Remove infected plants and practice crop rotation.',
            'prevention': 'Use disease-free seeds and transplants. Avoid overhead irrigation and working in fields when foliage is wet.'
        },
        'Tomato___Early_blight': {
            'symptoms': 'Dark brown to black lesions with concentric rings forming a "target spot" pattern. Usually begins on lower leaves.',
            'treatment': 'Apply fungicides preventatively. Remove and destroy infected leaves.',
            'prevention': 'Mulch around plants, provide adequate spacing, and avoid overhead watering.'
        },
        'Tomato___Late_blight': {
            'symptoms': 'Water-soaked, pale green to brown spots on leaves with white fuzzy growth on the underside in humid conditions. Can spread rapidly.',
            'treatment': 'Apply copper-based fungicides immediately. Remove infected plants to prevent spread.',
            'prevention': 'Plant resistant varieties, avoid overhead irrigation, and provide good air circulation.'
        },
        'Tomato___Leaf_Mold': {
            'symptoms': 'Pale green to yellow spots on upper leaf surfaces with olive-green to grayish-brown fuzzy mold on the undersides.',
            'treatment': 'Apply appropriate fungicides. Improve greenhouse ventilation if applicable.',
            'prevention': 'Improve air circulation, reduce humidity, and avoid leaf wetness.'
        },
        'Tomato___Septoria_leaf_spot': {
            'symptoms': 'Small, circular spots with dark borders and light gray centers. Black fruiting bodies may be visible in the center of spots.',
            'treatment': 'Apply fungicides when symptoms first appear. Remove infected leaves.',
            'prevention': 'Rotate crops, provide adequate spacing, use mulch, and avoid overhead irrigation.'
        },
        'Tomato___Spider_mites Two-spotted_spider_mite': {
            'symptoms': 'Tiny yellow or brown spots on leaves. Fine webbing on undersides of leaves. Leaves may become yellow, dry, and fall off when heavily infested.',
            'treatment': 'Apply miticides or insecticidal soap. Rinse plants with strong water spray to dislodge mites.',
            'prevention': 'Maintain plant vigor, increase humidity around plants, and introduce predatory mites.'
        },
        'Tomato___Target_Spot': {
            'symptoms': 'Circular brown spots with concentric rings giving a "target" appearance. Occurs on leaves, stems, and fruits.',
            'treatment': 'Apply appropriate fungicides. Prune affected areas and improve air circulation.',
            'prevention': 'Provide adequate spacing, avoid overhead irrigation, and remove infected plant debris.'
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'symptoms': 'Yellowing and upward curling of leaves, stunted growth, and flower drop. Plants may be severely stunted with no fruit production.',
            'treatment': 'No cure available. Remove and destroy infected plants to prevent spread.',
            'prevention': 'Use resistant varieties, control whitefly vectors with appropriate insecticides, and use reflective mulches.'
        },
        'Tomato___Tomato_mosaic_virus': {
            'symptoms': 'Mottled light and dark green pattern on leaves, distorted or small leaves, and yellow mottling.',
            'treatment': 'No cure available. Remove and destroy infected plants to prevent spread.',
            'prevention': 'Use disease-free seeds, resistant varieties, and sanitize tools and hands when working with plants.'
        },
        'Tomato___healthy': {
            'symptoms': 'No symptoms of disease. Leaves show normal color and structure.',
            'treatment': 'No treatment needed.',
            'prevention': 'Maintain good gardening practices such as proper spacing, watering at the base, and crop rotation.'
        }
    }
    
    # Default response if disease information is not available
    default_info = {
        'symptoms': 'Information not available.',
        'treatment': 'Consult a plant pathologist or agricultural extension service.',
        'prevention': 'Practice good garden hygiene and crop rotation.'
    }
    
    return disease_info.get(disease_class, default_info)

if __name__ == '__main__':
    app.run(debug=True)