import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# Load your trained models (consider loading them once to save resources)
count_model = YOLO('../models/count_best.pt')  # Model for counting
identify_model = YOLO('../models/identify_best.pt')  # Model for identifying

def process_image_count(image_path, conf_thresh):
    # Run inference with updated confidence threshold
    results = count_model([image_path], imgsz=640, conf=conf_thresh, show_labels=False, show_conf=False)

    # Number of detected colonies
    num_colonies = len(results[0].boxes.xyxy)

    # Load original image
    img = cv2.imread(image_path)

    # Draw bounding boxes
    for box in results[0].boxes.xyxy:
        start_point = (int(box[0]), int(box[1]))  # (x1, y1)
        end_point = (int(box[2]), int(box[3]))  # (x2, y2)
        color = (255, 0, 0)  # BGR color for the box, here it's red
        thickness = 2
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

    # Convert BGR to RGB for Streamlit display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, num_colonies

def process_image_identify(image_path, conf_thresh):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Convert BGR to RGB as YOLO expects RGB images
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run the model prediction
    results = identify_model(img_rgb, imgsz=640, conf=conf_thresh)

    species_counts = {}
    species_colors = {}
    unique_colors = np.random.randint(0, 255, size=(len(results[0].names), 3), dtype="uint8")

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())  # Convert class ID to integer
            species = result.names[class_id]  # Mapping class ID to species name
            conf = box.conf[0].item()  # Confidence score

            if conf < conf_thresh:
                continue

            species_counts[species] = species_counts.get(species, 0) + 1

            if species not in species_colors:
                species_colors[species] = unique_colors[class_id].tolist()

            color = species_colors[species]
            cords = box.xyxy[0].tolist()  # Coordinates of the bounding box
            start_point, end_point = (int(cords[0]), int(cords[1])), (int(cords[2]), int(cords[3]))
            img = cv2.rectangle(img, start_point, end_point, color, 2)

    img_rgb_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    legend_info = [(species, species_counts[species], species_colors[species]) for species in species_counts]

    return img_rgb_display, species_counts, legend_info

# UI Components
st.title('Colony Detection and Identification')

mode = st.radio("Choose Mode:", ('Count', 'Identify'))
conf_thresh = st.slider('Confidence Threshold:', 0.0, 1.0, 0.3)

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            fp = tmp_file.name
            tmp_file.write(uploaded_file.getvalue())

            if mode == 'Count':
                processed_img, num_colonies = process_image_count(fp, conf_thresh)
                st.image(processed_img, caption=f"Detected Colonies: {num_colonies}")
            else:  # Identify mode
                processed_img, species_counts, legend_info = process_image_identify(fp, conf_thresh)
                st.image(processed_img, caption="Detected Species")

                # Display species, counts, and color legend
                for species, count, color in legend_info:
                    color_rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
                    st.markdown(f"**{species}**: {count} <span style='color:{color_rgb};'>&#9608;</span>", unsafe_allow_html=True)
