
# Low-Light Image Enhancement Using TensorFlow and Keras

This project implements a low-light image enhancement model using a recursive residual group-based architecture inspired by MIRNet (Multi-Scale Residual Block-based Image Restoration Network). The model is trained on the LOL (Low-Light) dataset to enhance low-light images to their enhanced counterparts.

## Features
- **Low-Light Image Enhancement**: The model enhances images with poor lighting by learning mappings between low-light and enhanced images.
- **Multi-Scale Residual Blocks**: Uses advanced blocks for better feature extraction at multiple scales.
- **Attention Mechanisms**: Incorporates spatial and channel attention mechanisms to improve performance.

## Dataset

The dataset used in this project is the LOL (Low-Light) dataset, consisting of:
- Low-light images.
- Enhanced images.

You can download the dataset using the provided script.

## Dependencies

Make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow 2.x
- OpenCV
- Numpy
- Pillow
- Matplotlib
- gdown (for downloading the dataset)

To install the required Python packages, run:

```bash
pip install tensorflow opencv-python pillow matplotlib gdown
```

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/low_light_image_enhancement.git
   cd low_light_image_enhancement
   ```

2. **Download and unzip the LOL dataset**:
   The dataset is downloaded and extracted using the following commands:
   ```python
   !gdown https://drive.google.com/uc?id=1DdGIJ4PZPlF2ikl8mNM9V-PdVxVLbQi6
   !unzip -q lol_dataset.zip
   ```

3. **Training the Model**:
   The model is trained with low-light and enhanced image pairs from the dataset. To initiate training, simply run the script:
   ```python
   python train.py
   ```

4. **Data Visualization**:
   The dataset provides options to visualize low-light and enhanced images:
   ```python
   def afficher_images(images,titles):
       # Visualization code to display images
   ```

5. **Model Architecture**:
   - **Selective Kernel Feature Fusion (SKFF)**: Aggregates multi-scale features.
   - **Spatial and Channel Attention**: Enhances important spatial and channel features.
   - **Recursive Residual Blocks**: Captures hierarchical image features.
   - **Down and Up Sampling Modules**: Enables feature extraction at different resolutions.

6. **Metrics**:
   The model uses the following metrics for evaluation:
   - **PSNR (Peak Signal-to-Noise Ratio)**: Evaluates image reconstruction quality.
   - **MSE (Mean Squared Error)**: Loss function for training.

## Results

Training and validation loss, as well as PSNR values, are plotted after training for 50 epochs:

```python
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Losses Over Epochs")
plt.legend()
plt.grid()
plt.show()
```

## How to Run

1. Clone the repository and install the necessary dependencies.
2. Download the dataset and train the model.
3. Visualize the training process and evaluate the performance of the model on validation/test sets.


