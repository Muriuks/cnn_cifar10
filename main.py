from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI(title="Image Classification API")

#load model

model = load_model("cnn_cifar10.keras")

#cifar10 class labels
classes = ['airplane', 
           'automobile', 
           'bird', 
           'cat', 
           'deer',
           'dog',
           'frog', 
           'horse', 
           'ship', 
           'truck'
]

@app.get("/")
def root():
    return {"message": "Welcome to the Image Classification API"}

#prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))

    #convert to rgb if not already
    image=image.convert("RGB")
    
    # Resize the image to 32x32 pixels (CIFAR-10 input size)
    image = image.resize((32, 32))
    
    #conver to numpy array
    image_array = np.array(image)

    #normalize the pixel values to [0, 1]
    image_array = image_array.astype(np.float32) / 255.0
    
    # Expand dimensions to match the model's input shape (1, 32, 32, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction using the loaded model
    predictions = model.predict(image_array)
    
    #predictions with confidence level
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = classes[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])

    
    return {"predicted_class": predicted_class_name, "confidence": confidence}
