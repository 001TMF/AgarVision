from ultralytics import YOLO

# Load a model
model = YOLO('..models/count_best.pt')  # load a custom model

# Validate the model
def main():
    model.val(data="../data/dataset.yaml")  # train the model

if __name__ == '__main__':
    main()