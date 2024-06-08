from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch or use a allready trained model.

def main():
    # Assuming model is defined and configured earlier in the script
    model.train(data="./dataset/dataset.yaml", epochs=300, batch=16)  # train the model

if __name__ == '__main__':
    main()
