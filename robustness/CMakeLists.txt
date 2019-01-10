# minimum version of CMake
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(tracksparse)

set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")


find_package(OpenCV 3.4.3 REQUIRED)
IF (OpenCV_FOUND)
  message([message] OpenCV_VER:${OpenCV_VERSION})
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIBRARY_DIRS})
  add_definitions(${OpenCV_DEFINITIONS})
ENDIF()

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(
  robustness
  robustness.cpp
  )

target_link_libraries(
  robustness
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES} pcl_common
    
)
