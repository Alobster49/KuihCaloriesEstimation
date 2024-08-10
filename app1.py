import streamlit as st
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms

# Load the model
model_path = 'E:\\Code\\v6\\best.pt'  # Make sure this is the correct path to your .pt file
model = torch.load('E:\\Code\\v6\\yolov8s-seg.pt')
model.eval()  # Set the model to evaluation mode

# Define the classes and their corresponding calorie information
class_names = ['Kuih lompang', 'Kuih-Ketayap', 'karipap', 'kaswi', 'nona-manis']
calorie_info = {
    'Kuih lompang': 118,
    'Kuih-Ketayap': 140,
    'karipap': 82,
    'kaswi': 167,
    'nona-manis': 165
}

# Function to transform image
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize(640),  # Resize the image to the size your model expects
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to perform prediction
def predict(image):
    # Convert image to RGB and apply transformations
    image_tensor = transform_image(image.convert('RGB'))
    with torch.no_grad():  # No gradient needed for inference
        results = model(image_tensor)
        results = results.xyxy[0]  # Extract results
    return results

# Function to draw bounding boxes and labels on the image
def draw_boxes(results, image):
    draw = ImageDraw.Draw(image)
    for _, row in results.iterrows():
        bbox = row[['xmin', 'ymin', 'xmax', 'ymax']].tolist()
        class_id = int(row['class'])  # Extract class ID
        class_name = class_names[class_id]  # Get class name using ID
        score = row['confidence']
        calories = calorie_info.get(class_name, "Unknown")
        label = f"{class_name} {calories} Cal"
        draw.rectangle(bbox, outline='red', width=3)
        draw.text((bbox[0], bbox[1] - 10), f"{label} ({score:.2f})", fill='red')
    return image

# Streamlit interface
st.title('Malaysian Traditional Kuih Calories Estimation')
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Kuih', use_column_width=True)
    if st.button('Estimate Calories'):
        with st.spinner('Processing...'):
            results = predict(image)
            annotated_image = draw_boxes(results, image)
            st.image(annotated_image, caption='Processed Image', use_column_width=True)
