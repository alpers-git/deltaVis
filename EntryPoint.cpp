#include <stdio.h>

#include "GL/gl.h"
#include "GLFWHandler.h"
// our device-side data structures
#include "DeviceCode.h"
#include "Renderer.h"
#include <owl/owl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "owl/helper/cuda.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./../../../deltaVis/submodules/owl/3rdParty/stb_image/stb/stb_image_write.h"


const char *outFileName = "s01-simpleTriangles";

int main(int ac, char **av)
{
  // create a context on the first device:
  Renderer renderer;
  renderer.Init();

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################

  // create a window and a GL context
  auto glfw = GLFWHandler::getInstance();
  glfw->initWindow(renderer.fbSize.x, renderer.fbSize.y, "DeltaVisViewer");

  int fCount = 0;
  while(!glfw->windowShouldClose())
  {
    //render the frame
    renderer.Render();
    const uint32_t *fb
        = (const uint32_t*)owlBufferGetPointer(renderer.frameBuffer,0);

    float mouseDeltaX = glfw->mouseState.mouseDelta.x;
    float mouseDeltaY = glfw->mouseState.mouseDelta.y;

    if(glfw->mouseState.leftButtonDown)
    {
      renderer.camera.lens.center += renderer.camera.lens.du * mouseDeltaX;
      renderer.camera.lens.center += renderer.camera.lens.dv * mouseDeltaY;
      renderer.UpdateCamera();
    }
    
    //draw the frame to the window
    glfw->draw((void*)fb);

    assert(fb);
    
    // stbi_write_png(std::string(outFileName + std::to_string(fCount%10) + ".png").c_str(),renderer.fbSize.x,renderer.fbSize.y,4,
    //                 fb,renderer.fbSize.x*sizeof(uint32_t));
    fCount++;

    glfw->swapBuffers();
    glfw->pollEvents();
  }
  renderer.Shutdown();
  glfw->destroyWindow();

  return 0;
}