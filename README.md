# TTCDist: Fast Distance Estimation From an Active Monocular Camera Using Time-to-Contact

This repository contains code for *TTCDist: Fast Distance Estimation From an Active Monocular Camera Using Time-to-Contact* by Levi Burner, Nitin J. Sanket, Cornelia Fermuller, and Yiannis Aloimonos.

You can find a pdf of the paper [here](https://arxiv.org/abs/2203.07530).

[![TTCDistVideo](https://img.youtube.com/vi/CGS2FIZujnQ/0.jpg)](https://www.youtube.com/watch?v=CGS2FIZujnQ)

## Setup
We use standard scientific Python (`>=3.8`) libraries like NumPy, SciPy, Matplotlib, and Numba. Make sure these are installed.

The python bindings of the official [`apriltag`](https://github.com/AprilRobotics/apriltag) library are needed to use the `--april` flag in any of the scripts below.

The Python package `pyrealsense2` is needed to run the algorithms in realtime with a RealSense D435i camera. The following links may be helpful when tring to install `pyrealsense2` on Ubuntu 22.04 and newer. Older versions of Ubuntu should use the standard instructions.

* https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
* https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python

## Running from a recording in a folder
### Data
We provide the data for the 10 sequences considered in TTCDist [here](https://drive.google.com/file/d/1XPYIrPPVfR7nqbWSclclwrIp4igqRGyS/view?usp=sharing).

Download and extract to `TTCDist/recordings_21_11_12`

### Running on one sequence
```bash
python3 ttc_depth_from_folder.py --dir recordings_21_11_12/record_000000 --visualize
```

### Benchmark on a single sequence

*Note:* On the first run, the Numba functions will be compiled which will significantly affect the runtime. However, when run a second time, a compilation cache will be used instead.

```bash
python3 ttc_depth_from_folder.py --dir recordings_21_11_12/record_000000 --bench
```

### Benchmark on all sequences

*Note:* On the first run, the Numba functions will be compiled which will significantly affect the runtime. However, when run a second time, a compilation cache will be used instead.

```bash
python3 ttc_depth_from_folder.py --dir recordings_21_11_12 --all --bench
```

### Calculate Average Trajectory Error
We provide the data produced by all methods on the sequences in TTCDist [here](https://drive.google.com/file/d/1bO09vdTp8yazhRv755r_PQv2GPik6mY1/view?usp=sharing).

Download and extract to `TTCDist/results`

Then run the following to regenerate the results for the Tau, Phi, and AprilTag methods and calculate error.

```bash
python3 ttc_depth_from_folder.py --dir recordings_21_11_12 --all --save --april
python3 ttc_depth_calc_error.py --dir results --calc --latex
```

To use the `--april` flag the python bindings for the official apriltag package must be installed from [here](https://github.com/AprilRobotics/apriltag).

If you wish to reevaluate ROVIO and VINS-Mono on the recorded sequences the scripts in the `ros/` folder may be

## Running in realtime with an Intel Realsense D435i
### Without apriltag package
Assuming the `pyrealsense2` package is installed just run the following

```bash
python3 ttc_depth_realsense.py --visualize
```

A live output will run and the Z distance between the camera and the object will be shown. The top number is from the Tau constraint and the bottom is from the Phi constraint.

The `r` key can be used to reset the patch being tracked. The `q` key will close the application.

### With apriltag ground truth
If the python bindings for the official apriltag package are installed from [here](https://github.com/AprilRobotics/apriltag) then the following will work

```bash
python3 ttc_depth_realsense.py --visualize --april
```

in another terminal run


```bash
python3 ttc_depth_plot_live.py
```

A live plot will show. The top row shows the X, Y, and Z positions of the tracked object. `x_hat` is from the Tau Constraint. `x_gt` is from the AprilTag, and `phi_x_hat` is from the Phi Constraint.


## Running the RoboMaster experiments
Install the [RoboMaster-SDK Python package](https://github.com/dji-sdk/RoboMaster-SDK). Make sure your RoboMaster robot works with some of the examples included in the SDK. Then to run the experiments from TTCDist use the command:

```bash
python3 ttc_depth_robomaster.py --visualize --b 2.0 --efference
```

Remove the `--efference` flag to use measured acceleration as described in the paper.


## Acknowledgements
Part of the [rpg_trajectory_evaluation](https://github.com/uzh-rpg/rpg_trajectory_evaluation) toolbox is reproduced in `rpg_align_trajectory/` and is available under its respective license.
