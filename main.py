#مرتضى ضياء -- دعاء محمد 
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 10  # User can increase this
MODEL_PATH = 'brain_tumor_model.h5'
TRAINING_DIR = 'Training'
TESTING_DIR = 'tasting'  # User's specific folder name

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# --- Data Preparation ---
def load_data():
    """
    Loads data from both Training and tasting directories,
    merges them, and returns arrays + labels.
    """
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    X = []
    y = []
    
    # Helper to load from a directory
    def _load_from_dir(base_dir):
        if not os.path.exists(base_dir):
            print(f"Warning: Directory {base_dir} not found.")
            return
        
        for label in labels:
            path = os.path.join(base_dir, label)
            if not os.path.exists(path):
                continue
                
            class_num = labels.index(label)
            for img_name in os.listdir(path):
                try:
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(class_num)
                except Exception as e:
                    pass
    
    print("Loading data from Training directory...")
    _load_from_dir(TRAINING_DIR)
    print("Loading data from tasting directory...")
    _load_from_dir(TESTING_DIR)
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize
    X = X / 255.0
    
    return X, y, labels

# --- Model Definition ---
def create_model(num_classes):
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Global Model Variable ---
model = None
class_names = []

# --- Training / Loading ---
def initialize_system():
    global model, class_names
    
    # 1. Load Data
    X, y, labels = load_data()
    class_names = labels
    
    if len(X) == 0:
        print("No data found! Please check Training/tasting directories.")
        return

    # 2. Split Data (80% Training, 20% Testing) as requested
    X, y = shuffle(X, y, random_state=101)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
    print(f"Data Loaded. Training size: {len(X_train)}, Testing size: {len(X_test)}")
    
    # 3. Create or Load Model
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("Training new model...")
        model = create_model(len(class_names))
        history = model.fit(X_train, y_train, 
                            epochs=EPOCHS, 
                            validation_data=(X_test, y_test),
                            batch_size=BATCH_SIZE)
        model.save(MODEL_PATH)
        print("Model saved.")

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc*100:.2f}%")

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('mian.html')

@app.route('/analyze')
def analyze():
    return render_template('upload.html')

# --- Disease Info ---
DISEASE_INFO = {
    'glioma_tumor': "Glioma is a type of tumor that occurs in the brain and spinal cord. It begins in the gluey supportive cells (glial cells) that surround nerve cells and help them function.",
    'meningioma_tumor': "A meningioma is a tumor that arises from the meninges — the membranes that surround your brain and spinal cord. Most meningiomas are noncancerous (benign), though some can be cancerous.",
    'pituitary_tumor': "Pituitary tumors are abnormal growths that develop in your pituitary gland. Some pituitary tumors result in too much of the hormones that regulate important functions of your body.",
    'no_tumor': "No tumor detected. The scan appears normal, but please consult with a medical professional for a comprehensive evaluation."
}

# --- Tumor Type/Severity Classification ---
TUMOR_TYPE = {
    'glioma_tumor': 'malignant',      # Typically Malignant
    'meningioma_tumor': 'benign',     # Typically Benign
    'pituitary_tumor': 'benign',      # Typically Benign (or "tumor")
    'no_tumor': 'healthy'             # Healthy
}

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        # Save temp or process in memory
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Preprocess
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=0) # Add batch dimension
        img = img / 255.0
        
        # Predict
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        result_key = class_names[class_idx]
        
        # Format label specifically
        formatted_result = result_key.replace('_', ' ').title()
        
        # Get info message & Status
        message = DISEASE_INFO.get(result_key, "Information not available for this classification.")
        status_type = TUMOR_TYPE.get(result_key, 'unknown')

        # Override message if confidence is low or unknown? 
        # For now, just relying on the direct prediction.
        
        return jsonify({
            'prediction': formatted_result,
            'confidence': f"{confidence*100:.2f}%",
            'message': message,
            'status_type': status_type
        })

if __name__ == '__main__':
    # Initialize the AI system before starting server
    # Note: In a production server, this might be done differently
    try:
        initialize_system()
    except Exception as e:
        print(f"Initialization Error: {e}")
        # We continue to start the app, but prediction might fail if model isn't built
    
    print("Starting Flask Server...")
    app.run(debug=True, port=5000)