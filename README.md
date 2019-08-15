# AR-Projection-Mapping
This repository contains two implementations of AR Projection Mapping. In particular, it includes the calibration of the projector-camera pair using ArUco markers and the implementation of two versions of projection *tracking*.

### Installing
1. Install opencv and opencv-contrib.
2. 

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

### Camera Calibration (No Projector)


