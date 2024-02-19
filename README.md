# Cable Detection Using YoloV8
This code was made for SAUR to detect metal cables in cellulose bales.
It was made using a custom trained YOLOV8 model. Initially it was made to be used with the IFM O3R255 camera, the network was trained using a dataset obtained from this camera.
To use with other cameras its recomended to add more images to the dataset and train again.

## Functionality

The script uses a camera connected to the computer to capture real-time images. It detects metal cables in these images and display the distance between the centroid of the cables and a defined safety area.

The cable detection process follows these steps:

1. Capture of the image from the camera.
2. Send the image for the neural network to detection.
3. Process the outputs and give the distance in pixels.

## How to Use

To use the cable detector code, follow these steps:

1. Clone or download this repository to your computer
2. Ensure you have a realsense or ifm3d camera connected to the computer.
3. Setup the enviroment using the `setup.sh`: 

```
chmod +x setup.sh
./setup.sh
```
4. Choose wich code you want to run:
```
# Using ifm3d:
python3 ifm3d_detection.py

# Using images from a path:
python3 yolov8_custom.py
```

For using with other cameras its recomended to add more images to the dataset and train the network again.
The Jupyter Notebooks are used to train and test the neural network.