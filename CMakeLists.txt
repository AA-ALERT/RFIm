cmake_minimum_required(VERSION 3.8)
project(RFIm VERSION 1.0)
include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(TARGET_LINK_LIBRARIES OpenCL rt)

# FFlagger
add_executable(fflagger
  src/device.cpp
  src/filterbank.cpp
  src/opencl_error_handling.cpp
  src/main.cpp
)
target_include_directories(fflagger PRIVATE include)
target_link_libraries(fflagger PRIVATE ${TARGET_LINK_LIBRARIES})
install(TARGETS fflagger
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Unit testing
enable_testing()
set(TEST_LINK_LIBRARIES gtest gtest_main pthread)
# fflaggerTest
add_executable(fflaggerTest
  src/device.cpp
  src/filterbank.cpp
  src/opencl_error_handling.cpp
  test/tests.cpp
)
target_include_directories(fflaggerTest PRIVATE include)
target_link_libraries(fflaggerTest PRIVATE ${TARGET_LINK_LIBRARIES} ${TEST_LINK_LIBRARIES})
add_test(NAME fflaggerTest COMMAND fflaggerTest)
