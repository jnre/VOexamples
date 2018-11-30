##using symlink for 3rd party packages
using symlink to link eigen library such that i dont have to -I on compiling

go to /usr/local/include where g++ compiler defaults to finding header files

use sudo ln -sf /home/joseph/eigenfolder/Eigen /usr/local/include to link to include if u are not in include dir

now u can run eigen/eigen.h

note to self: eigen/unsupported is also added as symlink.

to remove symlink, either use rm or unlink

this only applies for usr/local/include where 3rd party software is installed, however since libeigen3 is installed at usr/include, easier to use cmake

##cmake

use cmake for further project such that CMakeLists.txt has include_directory("usr/include/eigen3")
add_executable( eigenMatrix eigenMatrix.cpp )
