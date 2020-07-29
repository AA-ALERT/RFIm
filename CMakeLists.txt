cmake_minimum_required(VERSION 3.8)
project(RFIm VERSION 1.0)
include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(TARGET_LINK_LIBRARIES rfim isa_utils isa_opencl astrodata OpenCL rt)
if($ENV{LOFAR})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_HDF5")
    set(TARGET_LINK_LIBRARIES ${TARGET_LINK_LIBRARIES} hdf5 hdf5_cpp z)
endif()
if($ENV{PSRDADA})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_PSRDADA")
    set(TARGET_LINK_LIBRARIES ${TARGET_LINK_LIBRARIES} psrdada cudart)
endif()

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

# RFImTesting
add_executable(RFImTesting
    src/RFImTesting.cpp
    include/configuration.hpp
    ${SNR_HEADER}
)
target_include_directories(RFImTesting PRIVATE include)
target_link_libraries(RFImTesting PRIVATE ${TARGET_LINK_LIBRARIES})

# RFImTuning
add_executable(RFImTuning
    src/RFImTuning.cpp
    include/configuration.hpp
    ${SNR_HEADER}
)
target_include_directories(RFImTuning PRIVATE include)
target_link_libraries(RFImTuning PRIVATE ${TARGET_LINK_LIBRARIES})

# Install
install(TARGETS rfim RFImTesting RFImTuning
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)