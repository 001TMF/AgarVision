"""from ultralytics import YOLO

# Load a model
model = YOLO('E:/BIO-ID/runs/detect/train30/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['E:/BIO-ID/test/4.png'], show_labels=False, show_conf=False)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk"""

import cv2
from ultralytics import YOLO

# Load a model
model = YOLO('../models/count_best.pt')
#model = YOLO('../models/identify_best.pt')

# Run batched inference on a list of images
results = model(['../outputs'], show_labels=False, show_conf=False)

# Process results list
for result in results:
    # Load original image
    img_path = 'path/to/images'  # Adjust if you're processing multiple images
    img = cv2.imread(img_path)

    # Number of detected colonies
    num_colonies = len(result.boxes.xyxy)

    # Iterate through boxes and draw them
    for box in result.boxes.xyxy:
        start_point = (int(box[0]), int(box[1]))  # (x1, y1)
        end_point = (int(box[2]), int(box[3]))  # (x2, y2)
        color = (255, 0, 0)  # BGR color for the box, here it's red
        thickness = 2
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

    # Prepare text
    text = f"Colonies: {num_colonies}"
    org = (50, 50)  # Top-right corner; adjust as needed
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)  # White color
    thickness = 2
    img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Save the image with the number of colonies indicated
    cv2.imwrite('result_with_count.jpg', img)

