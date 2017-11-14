
# 3D batch merge

## eclipse build tip
``` cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j8```


--------------------------------
This program executes Libviso, DLoop and g2o in parallel threads for trajectory estimation and 
optimization using stereo vision.

--------------------------------
Procedure -

1. Install

  1. Libviso2 - http://www.cvlibs.net/software/libviso/

  1. DLoop - https://github.com/dorian3d/DLoopDetector (along with DLib and DBow2)

  1. g2o - https://github.com/RainerKuemmerle/g2o

--------------------------------
Execute:

* cd cair_online/

* mkdir build

* cd build

* cmake ..

* make -j5

* ./main

--------------------------------

* Source code - main.cpp
* Supporting files - src/*.cpp
* Header files - includes/*.h

--------------------------------

Main function calls -

1. Libviso2 for intial trajectory estimates from src/helper.cpp - my_libviso2

1. DLoop Detector for loop closure from includes/demoDetector.h - run

1. g2o is called after every 500th frame and once at the end of trajectory.

--------------------------------

Parameters -

1. In main function
    1. IMG_DIR1 - Directory where Images are stored. Format -> dir/loop1/left and dir/loop1/right
    1. VOC_FILE - Vocabulary file. Generated from DBOW2
    1. IMAGE_W, IMAGE_H - Image Width, Image Height - Keep default.

1. In helperfunctions.cpp
     1. param.calib.f - focal length in pixels param.calib.cu - principal point (u-coordinate) in pixels param.calib.cv - principal point (v-coordinate) in pixels param.base - baseline in meters
  In demoDetector.h
     1. params.use_nss - use normalized similarity score instead of raw score params.alpha - nss threshold params.k - a loop must be consistent with 1 previous matches params.geom_check - use direct index for geometrical checking params.di_levels - use two direct index levels
  In TemplatedLoopDetector.h
     1. dislocal = number * f - for skipping 'number' of frames between loops.
