cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
project(deltaVis LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(SDL2 REQUIRED)
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

#--------------------ImGui---------------------#
# Set where the ImGui files are stored
set(IMGUI_PATH  submodules/imgui)

# Compile as static library 
file(GLOB IMGUI_SOURCES ${IMGUI_PATH}/*.cpp)
add_library("ImGui" STATIC ${IMGUI_SOURCES}
            ${IMGUI_PATH}/backends/imgui_impl_glfw.cpp
            ${IMGUI_PATH}/backends/imgui_impl_opengl3.cpp)
target_include_directories("ImGui" PUBLIC ${IMGUI_PATH} ${IMGUI_PATH}/backends glfw)
include_directories(submodules/imgui)
#---------------------------------------------#
#-----------------imgui-tfn-------------------#
set(IMGUI_TFN_PATH  submodules/imgui-transfer-function)

# Compile as static library
file(GLOB IMGUI_TFN_SOURCES ${IMGUI_TFN_PATH}/*.cpp ${IMGUI_TFN_PATH}/*.c)
add_library("ImGuiTFN" STATIC ${IMGUI_TFN_SOURCES})
target_include_directories("ImGuiTFN" PUBLIC ${IMGUI_TFN_PATH})
include_directories(submodules/imgui-transfer-function)
#---------------------------------------------#
#umesh
add_subdirectory(submodules/umesh EXCLUDE_FROM_ALL)
add_definitions(${UMESH_DEFINITIONS})
include_directories(${UMESH_INCLUDES})

include_directories(submodules/glfw/include)

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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")

add_executable(deltaVisViewer
    EntryPoint.cpp
    Renderer.h
    Renderer.cpp
    Renderer.cu
    GLFWHandler.h
    GLFWHandler.cpp
    Camera.h
    Camera.cpp
    CameraManipulator.h
    CameraManipulator.cpp
    UnstructuredElementHelpers.h
    stb_image_write.h
    # submodules/imgui-transfer-function/transfer_function_widget.cpp
    # submodules/imgui-transfer-function/gl_core_4_5.c
)
target_link_libraries(deltaVisViewer PRIVATE -lGL -lGLX)
target_link_libraries(deltaVisViewer 
    PRIVATE 
    deviceCode_ptx
    PUBLIC ${OPENGL_LIBRARIES}
    owl::owl
    glfw
    ImGui
    ImGuiTFN
    # imgui_tfn
    umesh)

