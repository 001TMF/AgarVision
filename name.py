import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# Load your trained model (consider loading it once to save resources)
model = YOLO('./best.pt')

def process_image(image_path):
    # Run inference
    results = model([image_path], imgsz=640, show_labels=False, show_conf=False)

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

st.title('Colony Detection App')

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            fp = tmp_file.name
            tmp_file.write(uploaded_file.getvalue())
            processed_img, num_colonies = process_image(fp)

            st.image(processed_img, caption=f"Detected Colonies: {num_colonies}")
