#include <stdio.h>

#include "GL/gl.h"
#include "GLFWHandler.h"
// our device-side data structures

#include "CameraManipulator.h"
#include "DeviceCode.h"
#include "Renderer.h"
#include <owl/owl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "owl/helper/cuda.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


const char *outFileName = "s01-simpleTriangles";

using namespace deltaVis;

int main(int ac, char **av)
{
  stbi_flip_vertically_on_write(true);
  // create a context on the first device:
  Renderer renderer;
  renderer.Init();

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################

  // create a window and a GL context
  auto glfw = GLFWHandler::getInstance();
  glfw->initWindow(renderer.fbSize.x, renderer.fbSize.y, "DeltaVisViewer");
  CameraManipulator controller = CameraManipulator(&renderer.camera);

  int fCount = 0;
  while(!glfw->windowShouldClose())
  {
    //render the frame
    renderer.Render();
    const uint32_t *fb
        = (const uint32_t*)owlBufferGetPointer(renderer.frameBuffer,0);

    //draw the frame to the window
    glfw->draw((void*)fb);

    assert(fb);
    
    // if(glfw->key.isPressed(GLFW_KEY_ESCAPE))
    //   glfw->setWindowShouldClose(true);

    //Taking a snapshot of the current frame
    if(glfw->key.isPressed(GLFW_KEY_1) && glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //!
      stbi_write_png(std::string("frame.png").c_str(),renderer.fbSize.x,renderer.fbSize.y,4,
                    fb,renderer.fbSize.x*sizeof(uint32_t));
    fCount++;

    glfw->swapBuffers();
    glfw->pollEvents();
    controller.ProcessEvents();
    renderer.OnCameraChange();
  }
  renderer.Shutdown();
  glfw->destroyWindow();

  return 0;
}