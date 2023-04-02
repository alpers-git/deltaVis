#include <stdio.h>
#include <chrono>

#include "GL/gl.h"
// our device-side data structures

#include "DeviceCode.h"
#include "Renderer.h"
#include <owl/owl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "owl/helper/cuda.h"

// for data importing data sets
#include "umesh/io/UMesh.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace deltaVis;

int main(int ac, char **av)
{
  // this is needed for stb_image_write to work properly
  stbi_flip_vertically_on_write(true);

  // read the input file from cmd line
  if (ac < 2)
  {
    std::cout << "Usage: DeltaVisViewer <input file>" << std::endl;
    return 0;
  }
  std::string inputFileName = av[1];
  std::cout << "loading " << inputFileName << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  auto umeshHdlPtr = umesh::io::loadBinaryUMesh(inputFileName);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
  std::cout << "found " << umeshHdlPtr->tets.size() << " tetrahedra" << std::endl;
  std::cout << "found " << umeshHdlPtr->pyrs.size() << " pyramids" << std::endl;
  std::cout << "found " << umeshHdlPtr->wedges.size() << " wedges" << std::endl;
  std::cout << "found " << umeshHdlPtr->hexes.size() << " hexahedra" << std::endl;
  std::cout << "found " << umeshHdlPtr->vertices.size() << " vertices" << std::endl;

  // create a window and a GL context
  auto glfw = GLFWHandler::getInstance();
  glfw->initWindow(800, 800, "DeltaVisViewer");

  // create a context on the first device:
  Renderer renderer(umeshHdlPtr);
  renderer.Init();

  renderer.Resize(vec2i(800, 800));
  int x, y;
  x = y = 800;

  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;

  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(glfw->getWindow(), true);
  ImGui_ImplOpenGL3_Init("#version 130");

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################

  int fCount = 0;
  while (!glfw->windowShouldClose())
  {
    // render the frame
    renderer.Render();
    const uint32_t *fb = (const uint32_t *)owlBufferGetPointer(renderer.frameBuffer, 0);

    // draw the frame to the window
    glfw->draw((void *)fb);

    assert(fb);

    // if(glfw->key.isPressed(GLFW_KEY_ESCAPE))
    //   glfw->setWindowShouldClose(true);

    // Taking a snapshot of the current frame
    if (glfw->key.isPressed(GLFW_KEY_1) && glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //!
      stbi_write_png(std::string("frame.png").c_str(), renderer.fbSize.x, renderer.fbSize.y, 4,
                     fb, renderer.fbSize.x * sizeof(uint32_t));
    fCount++;

    // glfw->setWindowSize(x, y);
    // renderer.Resize(owl::vec2i(x, y));
    // x = y++;


    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    //ImGui::Begin("fuck", 0, ImGuiWindowFlags_None);

    // Actual GUI code goes here
    ImGui::ShowDemoWindow();

    //ImGui::End();
    ImGui::Render();
    ImGuiIO &io = ImGui::GetIO();
    //glViewport(0, 0, (GLsizei)io.DisplaySize.x, (GLsizei)io.DisplaySize.y);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfw->swapBuffers();
    glfw->pollEvents();
    renderer.Update();
  }
  renderer.Shutdown();
  glfw->destroyWindow();

  return 0;
}