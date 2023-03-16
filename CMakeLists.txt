cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(deltaVis VERSION 1.0.4 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenGL REQUIRED)

add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)

include(embed_ptx)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#--------------------GLFW---------------------#
set(GLFW_BUILD_DOCS OFF CACHE STRING "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE STRING "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE STRING "" FORCE)
set(GLFW_INSTALL        OFF CACHE STRING "" FORCE)
add_subdirectory(submodules/glfw)
set_property(TARGET glfw PROPERTY FOLDER "submodules/GLFW3" )
#----------------------------------------------#

#umesh
add_subdirectory(submodules/umesh)
add_definitions(${UMESH_DEFINITIONS})
include_directories(${UMESH_INCLUDES}
  submodules/glfw/include)

embed_ptx(
  OUTPUT_TARGET
    deviceCode_ptx
  EMBEDDED_SYMBOL_NAMES
    deviceCode_ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    DeviceCode.cu
)


add_executable(deltaVisViewer
    EntryPoint.cpp
    Renderer.h
    Renderer.cpp
)
target_link_libraries(deltaVisViewer PRIVATE 
    deviceCode_ptx
    PRIVATE ${OPENGL_LIBRARIES}
    owl::owl
    PRIVATE glfw
    ${UMESH_LIBRARIES})


