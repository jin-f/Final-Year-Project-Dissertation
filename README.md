# Detecting and tracking edible objects in 3D space

### Getting started
It is recommended to use a conda environment for this project. Once conda is installed, install a new environment with the following command
```
conda env export > packages.yml
```
Next, install `pytorch` through the official pytorch website. https://pytorch.org/
Now install the Zed SDK. Note that the project requires a Zed 2i camera or similar Zed camera to function.
Install the Zed SDK from the official website: https://www.stereolabs.com/developers/release/

### Running tracker
A example object detection model is provided as seen in the project report which detects banana objects. In order to run the project do the following
```
python tracker.py
```
An alternative version of the tracker is also provided where the object detection model is also used as the feature extractor as described in project report.
```
python tracker_extract_features.py
```
This will start the python side of the tracker where the coordinates of the tracked object will be sent to local port `8051`. This port can then be read off in Unity.

### Running tracker in Unity
Create a basic scene custom to your own scenario/desire. The `UDPReceive.cs` script in the `Unity Scripts` folder can be attached to an empty game object in order to receive the live 3D coordinates that it reads off the local port `8051`. The `EdibleObjectMovement.cs` script can be attached to the game object which is representing the tracked food object in 3D space. Within this object we can attach the receiver object to the edible game object.

Once scene is played, the live coordinates can be seen in Unity as well as the game object movement corresponding to the edible object's movement.

### Running other applications
Within the scope of the project a script to convert the UECFOOD100 dataset to yolo format is also provided in the case the dataset is needed for training a custom detection model. This is in `UECFOOD100 to yolo.ipynb`

Each iteration that was went through in the project is also available and located in the `Iterations` folder.

### Training a custom detection model
Refer to official yolov5 page which recommends Roboflow and provides guides on training custom model.
https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
