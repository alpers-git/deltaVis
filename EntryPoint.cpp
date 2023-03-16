#include <stdio.h>
// our device-side data structures
#include "DeviceCode.h"
#include "Renderer.h"
#include <owl/owl.h>

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

  int fCount = 0;
  while(fCount++ <8)
  {
    renderer.Render();

    const uint32_t *fb
        = (const uint32_t*)owlBufferGetPointer(renderer.frameBuffer,0);
    assert(fb);
    
    stbi_write_png(std::string(outFileName + std::to_string(fCount) + ".png").c_str(),renderer.fbSize.x,renderer.fbSize.y,4,
                    fb,renderer.fbSize.x*sizeof(uint32_t));
  }
  renderer.Shutdown();

  return 0;
}