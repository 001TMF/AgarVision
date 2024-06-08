# AgarVision

## Introduction
AgarVision is a machine learning-powered tool designed to automate the counting of colonies on agar plates. This project aims to assist microbiologists and researchers by providing accurate, efficient, and reproducible colony counts, leveraging custom-trained models and a user-friendly interface.

## Features
- **Automatic Colony Detection:** Utilizes state-of-the-art machine learning models to detect and count colonies with high accuracy.
- **Streamlit Interface:** Offers an interactive UI for easy visualization and manipulation of results.
- **Customizable:** Supports training on custom datasets to improve accuracy specific to various colony types and agar backgrounds.

## Installation

### Prerequisites
Ensure you have Python 3.8 or later installed on your machine. You can download it from [Python's official website](https://www.python.org/downloads/).

### Clone the Repository
```bash
git clone https://github.com/001TMF/AgarVision.git
cd AgarVision
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
or

### Setting Up the Conda Environment

To create a Conda environment with all the necessary dependencies, follow these steps:

1. Clone the BindRMSD repository to your local machine:
   ```bash
   git clone https://github.com/001TMF/AgarVision.git
   cd AgarVision
   ```
2. Create the Conda environment from the environment.yaml file:
   ```bash
   conda env create -f environment.yaml #not yet finished 
   ```
3. Activate the environment:
   ```bash
   conda activate AgarVision
   ```
    
## Usage
To use AgarVision, follow these instructions:

```bash
# Launch the Streamlit Application and witness the magic of AgarVision
streamlit run streamlit/streamlit_ui.py
```
```bash
# Train the model on new data like a true data whisperer
python scripts/train.py
```
```bash
# Evaluate the model’s performance and pray for good metrics
python scripts/validate.py
```


## The Truth About Our Model
Let’s be real—our current model identifies colonies like a toddler identifies fine art. But fear not! We're training a new, robust version on an exciting dataset that promises to significantly improve its performance. Coming soon to a petri dish near you!


## Contributing

Contributions to BindRMSD are welcome and appreciated. To contribute:

   1. Fork the repository.
   2. Create a new branch (git checkout -b feature-branch). 
   3. Make your changes and commit them (git commit -am 'Add some feature').
   4. Push to the branch (git push origin feature-branch).
   5. Create a new Pull Request.