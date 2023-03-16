#include <stdio.h>

#include "GL/gl.h"
#include "GLFW/glfw3.h"
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

GLuint   fbTexture  {0};
cudaGraphicsResource_t cuDisplayTexture { 0 };

void draw(GLFWwindow *handle, const void *fbPointer, const vec2i &fbSize)
{
  glfwMakeContextCurrent(handle);
  if (false) {
    (cudaGraphicsMapResources(1, &cuDisplayTexture));

    cudaArray_t array;
    (cudaGraphicsSubResourceGetMappedArray(&array, cuDisplayTexture, 0, 0));
    {
      cudaMemcpy2DToArray(array,
                          0,
                          0,
                          reinterpret_cast<const void *>(fbPointer),
                          fbSize.x * sizeof(uint32_t),
                          fbSize.x * sizeof(uint32_t),
                          fbSize.y,
                          cudaMemcpyDeviceToDevice);
    }
  } else {
    (glBindTexture(GL_TEXTURE_2D, fbTexture));
    glEnable(GL_TEXTURE_2D);
    (glTexSubImage2D(GL_TEXTURE_2D,0,
                              0,0,
                              fbSize.x, fbSize.y,
                              GL_RGBA, GL_UNSIGNED_BYTE, fbPointer));
  }

  glDisable(GL_LIGHTING);
  glColor3f(1, 1, 1);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, fbTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, fbSize.x, fbSize.y);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

  glBegin(GL_QUADS);
  {
    glTexCoord2f(0.f, 0.f);
    glVertex3f(0.f, 0.f, 0.f);

    glTexCoord2f(0.f, 1.f);
    glVertex3f(0.f, (float)fbSize.y, 0.f);

    glTexCoord2f(1.f, 1.f);
    glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

    glTexCoord2f(1.f, 0.f);
    glVertex3f((float)fbSize.x, 0.f, 0.f);
  }
  glEnd();
  if (false) {
    (cudaGraphicsUnmapResources(1, &cuDisplayTexture));
  }
}

int main(int ac, char **av)
{
  // create a context on the first device:
  Renderer renderer;
  renderer.Init();



  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  //set up glfw window
   if (!glfwInit())
        exit(EXIT_FAILURE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_VISIBLE, true);
  GLFWwindow *handle = glfwCreateWindow(renderer.fbSize.x, renderer.fbSize.y,
                                "DeltaVisViewer", NULL, NULL);
  if (!handle) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  //glfwSetWindowUserPointer(handle, this);
  glfwMakeContextCurrent(handle);
  glfwSwapInterval(0);
  glGenTextures(1, &fbTexture);
  (glBindTexture(GL_TEXTURE_2D, fbTexture));
  (glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, renderer.fbSize.x, renderer.fbSize.y, 0, GL_RGBA,
                            GL_UNSIGNED_BYTE, nullptr));
  int fCount = 0;
  while(!glfwWindowShouldClose(handle))
  {
    renderer.Render();
    const uint32_t *fb
        = (const uint32_t*)owlBufferGetPointer(renderer.frameBuffer,0);
    draw(handle, (void*)fb, renderer.fbSize);

    assert(fb);
    
    // stbi_write_png(std::string(outFileName + std::to_string(fCount%10) + ".png").c_str(),renderer.fbSize.x,renderer.fbSize.y,4,
    //                 fb,renderer.fbSize.x*sizeof(uint32_t));
    fCount++;

    //glfwMakeContextCurrent(handle);
    glfwSwapBuffers(handle);
    glfwPollEvents();
  }
  renderer.Shutdown();

  return 0;
}