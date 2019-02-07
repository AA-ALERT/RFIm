cmake_minimum_required(VERSION 3.8)
project(RFIm VERSION 0.1)
include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(TARGET_LINK_LIBRARIES OpenCL rt)

set(RFIM_HEADER
    include/RFIm.hpp
)

#librfim
add_library(rfim SHARED
    src/RFIm.cpp
)
set_target_properties(rfim PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "include/RFIm.hpp"
)
target_include_directories(rfim PRIVATE include)