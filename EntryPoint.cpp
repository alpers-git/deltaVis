#include <stdio.h>
#include <chrono>

// #include "GL/gl.h"

#define TFN_WIDGET_NO_STB_IMAGE_IMPL
#include "transfer_function_widget.h"

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

#define HEADLESS 0

using namespace deltaVis;

int main(int ac, char **av)
{
  // this is needed for stb_image_write to work properly
  stbi_flip_vertically_on_write(true);

  // read the input file from cmd line
  std::vector<owl::vec4f> colorMapVec;
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

  // create a context on the first device:
  Renderer renderer(umeshHdlPtr);
  renderer.Init();

  if (ac > 2)
  {
    // parse the argument that is the camera comfigurations -position, lookat, up, fovy, focalDist

    // find the argument after 1 that starts with -cam
    int camArg = 2;
    while (camArg < ac)
    {
      if (av[camArg][0] == '-' && av[camArg][1] == 'c' && av[camArg][2] == 'a' && av[camArg][3] == 'm')
      {
        break;
      }
      camArg++;
    }
    if (camArg < ac)
    {
      // parse the camera arguments
      vec3f lookFrom, lookAt, lookUp;
      float cosFovy;
      if (camArg + 4 < ac)
      {
        lookFrom = vec3f(atof(av[camArg + 1]), atof(av[camArg + 2]), atof(av[camArg + 3]));
        lookAt = vec3f(atof(av[camArg + 4]), atof(av[camArg + 5]), atof(av[camArg + 6]));
        lookUp = vec3f(atof(av[camArg + 7]), atof(av[camArg + 8]), atof(av[camArg + 9]));
        cosFovy = atof(av[camArg + 10]);
      }
      renderer.camera.setOrientation(lookFrom, lookAt, lookUp, toDegrees(cosFovy));
    }

    // find the argument after 1 that starts with -tf
    int tfArg = 2;
    while (tfArg < ac)
    {
      if (av[tfArg][0] == '-' && av[tfArg][1] == 't' && av[tfArg][2] == 'f')
      {
        break;
      }
      tfArg++;
    }
    if (tfArg < ac)
    {
      // get the path to the transfer function and parse the text file
      std::string tfPath = av[tfArg + 1];
      FILE *fp = fopen(tfPath.c_str(), "r");
      if (fp == NULL)
      {
        std::cout << "could not tf file " << tfPath << std::endl;
        return 0;
      }
      char line[1024];
      //first lien is opacity scale
      fgets(line, 1024, fp);
      float opacityScale = atof(line);
      renderer.SetOpacityScale(opacityScale);
      //parse the file till the end push vec4f into the colorMapVec
      while (fgets(line, 1024, fp) != NULL)
      {
        float r, g, b, a;
        sscanf(line, "%f %f %f %f", &r, &g, &b, &a);
        colorMapVec.push_back(vec4f(r, g, b, a));
      }
      renderer.SetColorMap(colorMapVec);
    }
    // find the argument that is -shadow
    int shadowArg = 2;
    while (shadowArg < ac)
    {
      if (av[shadowArg][0] == '-' && av[shadowArg][1] == 's' &&
       av[shadowArg][2] == 'h' && av[shadowArg][3] == 'a' &&
        av[shadowArg][4] == 'd' && av[shadowArg][5] == 'o' &&
         av[shadowArg][6] == 'w')
      {
        renderer.shadows = true;
        break;
      }
      shadowArg++;
    }
    //get three float values set to light direction
    if (shadowArg < ac)
    {
      renderer.SetLightDir(vec3f(atof(av[shadowArg + 1]),
       atof(av[shadowArg + 2]), atof(av[shadowArg + 3])));
    }
  }

// create a window and a GL context
#if HEADLESS
  auto glfw = GLFWHandler::getInstance();
  glfw->initWindow(1000, 1000, "DeltaVisViewer");
#endif

  renderer.Resize(vec2i(1000, 1000));
  int x, y;
  x = y = 1000;

#if HEADLESS
  //----------Create ImGui Context----------------
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  // init ImGui
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(glfw->getWindow(), true);
  ImGui_ImplOpenGL3_Init("#version 130");

  // init OpenGL for imgui tnf editor
  if (ogl_LoadFunctions() == ogl_LOAD_FAILED)
  {
    std::cerr << "Failed to initialize OpenGL\n";
    return 1;
  }
  // create a transfer function editor
  tfnw::TransferFunctionWidget tfn_widget;
#else
  ImGui::CreateContext();
  tfnw::TransferFunctionWidget tfn_widget;
  auto cm = tfn_widget.get_colormapf();
  if(colorMapVec.size() == 0)
  {
    printf("using default colormap\n");
    for (int i = 0; i < cm.size(); i += 4)
    {
      colorMapVec.push_back(owl::vec4f(cm[i],
                                      cm[i + 1], cm[i + 2], cm[i + 3]));
    }
  }
  renderer.SetColorMap(colorMapVec);
#endif

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  int fCount = 50;
  float avgFrameTime = 0.f;
#if HEADLESS
  while (!glfw->windowShouldClose())
#else
  while (fCount > 0)
#endif
  {
    //----------------Renderer and Windowing----------------
    // render the frame
    renderer.Render(HEADLESS == 0);
    const uint32_t *fb = (const uint32_t *)owlBufferGetPointer(renderer.frameBuffer, 0);
#if HEADLESS
    // draw the frame to the window
    glfw->draw((void *)fb);
    assert(fb);

    // Taking a snapshot of the current frame
    if (glfw->key.isPressed(GLFW_KEY_1) && glfw->key.isDown(GLFW_KEY_RIGHT_SHIFT)) //!
      stbi_write_png(std::string("frame.png").c_str(), renderer.fbSize.x, renderer.fbSize.y, 4,
                     fb, renderer.fbSize.x * sizeof(uint32_t));
    // Printing camera parameters as a callable arguments
    if (glfw->key.isPressed(GLFW_KEY_C))
      printf(" -cam %f %f %f %f %f %f %f %f %f %f\n",
             renderer.camera.position.x, renderer.camera.position.y,
             renderer.camera.position.z, renderer.camera.getAt().x,
             renderer.camera.getAt().y, renderer.camera.getAt().z,
             renderer.camera.getUp().x, renderer.camera.getUp().y,
             renderer.camera.getUp().z, renderer.camera.getCosFovy());
    //writing transferfunction vector as a png
    if(glfw->key.isPressed(GLFW_KEY_T))
    {
      auto cm = tfn_widget.get_colormapf();
      std::vector<owl::vec4f> colorMapVec;
      for (int i = 0; i < cm.size(); i += 4)
      {
        colorMapVec.push_back(owl::vec4f(cm[i],
                                         cm[i + 1], cm[i + 2], cm[i + 3]));
      }
      //create file and write to it
      FILE *fp = fopen("transferfunction.tf", "w");
      fprintf(fp, "%f\n", renderer.opacityScale);
      for(int i = 0; i < colorMapVec.size(); i++)
      {
        fprintf(fp, "%f %f %f %f\n", colorMapVec[i].x, colorMapVec[i].y, colorMapVec[i].z, colorMapVec[i].w);
      }
      //write opacity scale
      fclose(fp);
      printf("Transferfunction written to transferfunction.tf\n");
    }

    //----------------ImGui----------------
    // request new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Renderer Control Panel");
    if (ImGui::CollapsingHeader("Transfer Function Editor", ImGuiTreeNodeFlags_DefaultOpen))
    {
      static float opacity = 1.0f;
      ImGui::Text("Opacity scale");
      ImGui::SameLine();
      if (ImGui::SliderFloat("##5", &opacity, 0.0f, 1.0f))
        renderer.SetOpacityScale(opacity);
      if (ImGui::DragFloat("volume dt", &renderer.dt, 0.0001f, 0.0001f, 2.0f))
      {
        renderer.accumID = 0;
        renderer.dt = max(renderer.dt, 0.0001f);
      }
      if (ImGui::Checkbox("Shadows", &renderer.shadows))
        renderer.accumID = 0;
      ImGui::SameLine();
      if (ImGui::DragFloat3("Light direction", &renderer.lightDir[0], 0.01f, -1.0f, 1.0f))
        renderer.SetLightDir(renderer.lightDir);
      ImGui::TextColored(ImVec4(1, 1, 0, 1), "Avg. FPS: %.1f", 1.0f / renderer.avgFrameTime);
      tfn_widget.draw_ui();
    }
    ImGui::End();

    ImGui::Render(); // render frame

    // check if ImGui is capturing the mouse
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse) // then we don't want to poll the mouse from other apps
      glfw->mouseState.imGuiPolling = true;
    else
      glfw->mouseState.imGuiPolling = false;
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // poll tfn changes
    if (tfn_widget.changed())
    {
      auto cm = tfn_widget.get_colormapf();
      std::vector<owl::vec4f> colorMapVec;
      for (int i = 0; i < cm.size(); i += 4)
      {
        colorMapVec.push_back(owl::vec4f(cm[i],
                                         cm[i + 1], cm[i + 2], cm[i + 3]));
      }
      renderer.SetColorMap(colorMapVec);
    }

    glfw->swapBuffers();
    glfw->pollEvents();
#else
    fCount--;
    avgFrameTime += renderer.lastFrameTime;
    if (fCount == 0 || fCount == 45)
      stbi_write_png(std::string("frame_" + std::to_string(fCount) + ".png").c_str(), renderer.fbSize.x, renderer.fbSize.y, 4,
                     fb, renderer.fbSize.x * sizeof(uint32_t));
#endif
    renderer.Update(HEADLESS == 0);
  }
  printf("==Overall average frame time: %fs OR %fFPS ==\n\tgrid:%d %d %d\n\tdt:%f\n",
   avgFrameTime / 50.0f, 1.0f / (avgFrameTime / 50.0f), renderer.macrocellDims.x,
    renderer.macrocellDims.y, renderer.macrocellDims.z, renderer.dt);
  renderer.Shutdown();
#if HEADLESS
  glfw->destroyWindow();
#endif

  return 0;
}