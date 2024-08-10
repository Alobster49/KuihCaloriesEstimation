from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_image_comparison import image_comparison
import random

st.title("MALAYSIAN TRADITIONAL KUIH CALORIES ESTIMATION USING DEEP LEARNING 🤖🍘🍙")

st.sidebar.image('E:\\Code\\v6\\download (2).jpeg')

confidence_threshold = st.sidebar.slider("Set Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

model = YOLO('E:\\Code\\v6\\best.pt')

color_map = {
    'kaswi': (176, 58, 46),
    'karipap': (0, 50, 0),
    'kuih-ketayap': (52, 73, 94),
    'nona-manis': (190, 118, 189),
    'kuih lompang': (0, 255, 255)
}

calorie_ranges = {
    'kaswi': (162, 172),
    'karipap': (77, 87),
    'kuih-ketayap': (135, 145),
    'nona-manis': (160, 170),
    'kuih lompang': (113, 193)
}

weight_ranges = {
    'kaswi': (29, 39),
    'karipap': (65, 75),
    'kuih-ketayap': (50, 60),
    'nona-manis': (30, 40),
    'kuih lompang': (25, 35)
}

def get_calculation_calories(kuih, weight):
    calorie_range = calorie_ranges[kuih]
    weight_range = weight_ranges[kuih]
    weight_index = (weight - weight_range[0]) / (weight_range[1] - weight_range[0])
    calories = calorie_range[0] + weight_index * (calorie_range[1] - calorie_range[0])
    return round(calories)

def draw_masks(results, img_shape):
    height, width = img_shape[:2]
    background = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    for result in results:
        masks = result.masks.xy
        for mask in masks:
            mask = mask.astype(int)
            cv2.drawContours(background, [mask], -1, (0, 255, 0), thickness=cv2.FILLED)
    
    return background

uploaded_file = st.file_uploader("Upload an image for analysis", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_display = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    results = model(img, conf=confidence_threshold)
    img_with_boxes = img.copy()

    names_list = []
    calorie_counts = []
    weights_list = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confidences):
            r = box.xyxy[0].astype(int)
            predicted_name = result.names[int(box.cls[0])].lower()
            box_color = color_map.get(predicted_name, (255, 255, 255))

            cv2.rectangle(img_with_boxes, (r[0], r[1]), (r[2], r[3]), box_color, 2)
            cv2.putText(img_with_boxes, f"{predicted_name} ({conf*100:.2f}%)", (r[0], r[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            random_weight = random.randint(*weight_ranges[predicted_name])
            weights_list.append(random_weight)
            calories = get_calculation_calories(predicted_name, random_weight)
            calorie_counts.append(calories)
            names_list.append(predicted_name)

    mask_background = draw_masks(results, img.shape)

    combined_image = img_with_boxes.copy()
    combined_image[mask_background != 255] = mask_background[mask_background != 255]
    combined_image_display = Image.fromarray(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))

    st.markdown('##### Slider of Actual Image and Segmentation ')
    
    image_comparison(
       img1=img_display,
       img2=combined_image_display,
       label1="Actual Image",
       label2="Segmentation",
       width=700,
       starting_position=50,
       show_labels=True,
       make_responsive=True,
       in_memory=True
    )

    df_x = pd.DataFrame({
        'Names': names_list,
        'Weight (g)': weights_list,
        'Calories': calorie_counts
    })
    summary_table = df_x.groupby('Names').agg({'Weight (g)': 'sum', 'Calories': ['count', 'sum']}).reset_index()
    summary_table.columns = ['Type of Kuih 🍘', 'Total Weight (g)', 'Counts', 'Total Calories (Kcal)']
    total_calories = summary_table['Total Calories (Kcal)'].sum()
    
    st.sidebar.markdown('<h1 style="color:white;">Calories Estimation 🧮🤖</h1>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <style>
    .dataframe {
        font-size: large;
    }
    .dataframe th, .dataframe td {
        font-size: large;
    }
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.dataframe(summary_table)
    st.sidebar.markdown(f'<h3 style="color:white;">Total Calories 🚀: {total_calories} kcal</h3>', unsafe_allow_html=True)
    
    def calculate_time(activity_rate):
        total_minutes = total_calories / activity_rate
        hours, minutes = divmod(total_minutes, 60)
        return total_minutes, hours, minutes
    
    walking_time, walking_hours, walking_minutes = calculate_time(3.66)
    jogging_time, jogging_hours, jogging_minutes = calculate_time(8.53)
    swimming_time, swimming_hours, swimming_minutes = calculate_time(11.64)
    biking_time, biking_hours, biking_minutes = calculate_time(6.74)
    
    st.sidebar.markdown(f'#### Calorie Burn Rate for Common Activities🔥')
    st.sidebar.markdown(f'- Walking 🚶‍♂️: {walking_hours:.0f} hours {walking_minutes:.0f} minutes')
    st.sidebar.markdown(f'- Jogging 🏃‍♂️: {jogging_hours:.0f} hours {jogging_minutes:.0f} minutes')
    st.sidebar.markdown(f'- Swimming 🏊‍♂️: {swimming_hours:.0f} hours {swimming_minutes:.0f} minutes')
    st.sidebar.markdown(f'- Riding Bike 🚴‍♂️: {biking_hours:.0f} hours {biking_minutes:.0f} minutes')
else:
    st.error("Please upload an image file to proceed.")
