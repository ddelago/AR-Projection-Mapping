# AR-Projection-Mapping
This repository contains two implementations of AR Projection Mapping. In particular, it includes the calibration of the projector-camera pair using ArUco markers and the implementation of two versions of projection *tracking*.

### Installing
1. Project was built using a VM running Ubuntu 18.04
2. Install opencv-contrib.
    - `pip install opencv-contrib-python`
2. `git clone https://github.com/ddelago/AR-Projection-Mapping.git`

### Generating ArUco Calibration Markers
There are three ways to generate calibration markers. 
1. Single Marker
    - Generate a single ArUco marker. Can choose specific marker ID and image size.
    - `python generate_aruco.py`
2. ArUco Grid
    - Generates a grid of ArUco markers. 
    - `python generate_arucoGrid.py`
3. ChArUco Grid
    - Generates a chessboard filled with ArUco markers. This is the ideal method to use when calibrating.
    - `python generate_ChAruco.py`  

### Calibration
1. The first calibration method involves using images of a ChArUco board in various positions. 
    - `python calibration_ChAruco.py`
2. The second calibration method is a **stereo** calibration method that uses a combination of a physical ChArUco board along with a projected circle grid. 
    - See the ChAruco_Circles.webm calibration video to view an example.
    - Based off the calibration method [here](https://www.morethantechnical.com/2017/11/17/projector-camera-calibration-the-easy-way/).
    - This method treats the projector as the second camera in a stereo camera pair in order to get the transformation from the actual camera to the projectors perspective.
    - Needs a circle grid as well as a ChArUco board. To generate, use the `~/patterns/gen_pattern.py` file. Instructions on how to use the program can be seen [here](https://docs.opencv.org/master/da/d0d/tutorial_camera_calibration_pattern.html). An already generated board can be found here: `~/patterns/test_circleGrid`
    - `python calibration_ChArucoWithCircles.py`

