#include "Renderer.h"

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

const int NUM_VERTICES = 8;
vec3f vertices[NUM_VERTICES] =
  {
    { -1.f,-1.f,-1.f },
    { +1.f,-1.f,-1.f },
    { -1.f,+1.f,-1.f },
    { +1.f,+1.f,-1.f },
    { -1.f,-1.f,+1.f },
    { +1.f,-1.f,+1.f },
    { -1.f,+1.f,+1.f },
    { +1.f,+1.f,+1.f }
  };

const int NUM_INDICES = 12;
vec3i indices[NUM_INDICES] =
  {
    { 0,1,3 }, { 2,3,0 },
    { 5,7,6 }, { 5,6,4 },
    { 0,4,5 }, { 0,5,1 },
    { 2,3,7 }, { 2,7,6 },
    { 1,5,7 }, { 1,7,3 },
    { 4,0,2 }, { 4,2,6 }
  };


OWLVarDecl rayGenVars[] = {
        //framebuffer
        { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
        { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
        { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
        //camera
        { "camera.org",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.origin)},
        { "camera.llc",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.lower_left_corner)},
        { "camera.horiz",  OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.horizontal)},
        { "camera.vert",   OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.vertical)},
        //Volume data
        { "volume.elementTLAS", OWL_GROUP,     OWL_OFFSETOF(RayGenData,volume.elementTLAS) },
        { /* sentinel to mark end of list */ }
    };
namespace deltaVis
{
Renderer::Renderer(std::shared_ptr<umesh::UMesh> umesh)
  : umeshPtr(umesh)
{

}

Renderer::~Renderer()
{
}

void Renderer::Init()
{
    // create a context on the first device:
    context = owlContextCreate(nullptr,1);
    module = owlModuleCreate(context,deviceCode_ptx);

    // ##################################################################
    // set up all the *GEOMETRY* graph we want to render
    // ##################################################################

    // -------------------------------------------------------
    // declare geometry type
    // -------------------------------------------------------
   // Different intersection programs for different element types
    OWLVarDecl unstructuredElementVars[] = {
      { "tetrahedra", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, tetrahedra)},
      { "pyramids", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, pyramids)},
      { "hexahedra", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, hexahedra)},
      { "wedges", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, wedges)},
      { "bytesPerIndex", OWL_UINT, OWL_OFFSETOF(UnstructuredElementData, bytesPerIndex)},
      { "vertices", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, vertices)},
      { "scalars", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, scalars)},
      { "offset", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, offset)},
      { "numTetrahedra", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numTetrahedra)},
      { "numPyramids", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numPyramids)},
      { "numWedges", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numWedges)},
      { "numHexahedra", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numHexahedra)},
      { "maxima", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, maxima)},
      { "bboxes", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, bboxes)},
      {/* sentinel to mark end of list */}
    };

    OWLVarDecl triangleVars[] = {
      { "vertices", OWL_BUFPTR, OWL_OFFSETOF(TriangleData, vertices) },
      { "indices", OWL_BUFPTR, OWL_OFFSETOF(TriangleData, indices) },
      { "color", OWL_FLOAT3, OWL_OFFSETOF(TriangleData, color)},
      {}
    };
    
  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");
  frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

  // -----------------------------------------------------------------
  // Volume elements
  // -----------------------------------------------------------------
  tetrahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->tets.size() * 4, nullptr);
  pyramidsData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->pyrs.size() * 5, nullptr);
  wedgesData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->wedges.size() * 6, nullptr);
  hexahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->hexes.size() * 8, nullptr);
  verticesData = owlDeviceBufferCreate(context, OWL_FLOAT3, umeshPtr->vertices.size(), nullptr);
  scalarData = owlDeviceBufferCreate(context, OWL_FLOAT, umeshPtr->perVertex->values.size(), nullptr);
  owlBufferUpload(tetrahedraData, umeshPtr->tets.data());
  owlBufferUpload(pyramidsData, umeshPtr->pyrs.data());
  owlBufferUpload(wedgesData, umeshPtr->wedges.data());
  owlBufferUpload(hexahedraData, umeshPtr->hexes.data());
  owlBufferUpload(verticesData, umeshPtr->vertices.data());
  owlBufferUpload(scalarData, umeshPtr->perVertex->values.data());

  tetrahedraType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
  pyramidType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
  wedgeType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
  hexahedraType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
  triangleType = owlGeomTypeCreate(context, OWL_GEOMETRY_TRIANGLES, sizeof(TriangleData), triangleVars, -1);

  owlGeomTypeSetClosestHit(triangleType, 0, module, "TriangleClosestHit");

  // ------------------------------------------------------------------
  // triangle mesh
  // ------------------------------------------------------------------
  trianglesGeom = owlGeomCreate(context, triangleType);
  indexBuffer = owlDeviceBufferCreate(context, OWL_INT3, NUM_INDICES, indices);
  vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, NUM_VERTICES, vertices);
  owlTrianglesSetIndices(trianglesGeom, indexBuffer, NUM_INDICES, sizeof(vec3i), 0);
  owlTrianglesSetVertices(trianglesGeom, vertexBuffer, NUM_VERTICES, sizeof(vec3f), 0);
  owlGeomSetBuffer(trianglesGeom, "indices", indexBuffer);
  owlGeomSetBuffer(trianglesGeom, "vertices", vertexBuffer);
  owlGeomSet3f(trianglesGeom,"color", owl3f{0,1,0});

  // ------------------------------------------------------------------
  // the group/accel for that mesh
  // ------------------------------------------------------------------
  trianglesGroup = owlTrianglesGeomGroupCreate(context, 1, &trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);
  world = owlInstanceGroupCreate(context, 1, &trianglesGroup);
  owlGroupBuildAccel(world);
  //owlParamsSetGroup(lp, "trianglesTLAS", trianglesTLAS);



  // ##################################################################
  // set miss and raygen program required for SBT
  // ##################################################################

  // -------------------------------------------------------
  // set up miss prog
  // -------------------------------------------------------
  OWLVarDecl missProgVars[]
    = {
    { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color0)},
    { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color1)},
    { /* sentinel to mark end of list */ }
  };
  // ----------- create object  ----------------------------
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);

  // ----------- set variables  ----------------------------
  owlMissProgSet3f(missProg,"color0",owl3f{.2f,.2f,.26f});
  owlMissProgSet3f(missProg,"color1",owl3f{.1f,.1f,.16f});


    // -------------------------------------------------------
    // set up ray gen program
    // -------------------------------------------------------
    rayGen
        = owlRayGenCreate(context,module,"simpleRayGen",
                        sizeof(RayGenData),
                        rayGenVars,-1);
                        
    // ----------- set variables  ----------------------------
    owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
    owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
    owlRayGenSetGroup (rayGen,"world",        world);

    //set up camera controller
    controller = new CameraManipulator(&camera);
    OnCameraChange();

    // ##################################################################
    // build *SBT* required to trace the groups
    // ##################################################################
    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
}

void Renderer::Render()
{
    owlBuildSBT(context);
    owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
        // for host pinned mem it doesn't matter which device we query...
        //const uint32_t *fb = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
}

void Renderer::Update()
{
    if (controller->ProcessEvents())
      OnCameraChange();

    auto glfw = GLFWHandler::getInstance();
    if(glfw->getWindowSize() != fbSize)
      Resize(glfw->getWindowSize());
}

void Renderer::Shutdown()
{
    LOG("destroying devicegroup ...");
    owlContextDestroy(context);
}

void Renderer::Resize(const vec2i newSize)
{
    fbSize = newSize;
    owlBufferResize(frameBuffer,fbSize.x*fbSize.y);
    owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
    OnCameraChange();
}

void Renderer::OnCameraChange()
{
  const vec3f lookFrom = camera.getFrom();
  const vec3f lookAt = camera.getAt();
  const vec3f lookUp = camera.getUp();
  const float cosFovy = camera.getCosFovy();
  const float vfov = toDegrees(acosf(cosFovy));
  // ........... compute variable values  ..................
  const vec3f vup = lookUp;
  const float aspect = fbSize.x / float(fbSize.y);
  const float theta = vfov * ((float)M_PI) / 180.0f;
  const float half_height = tanf(theta / 2.0f);
  const float half_width = aspect * half_height;
  const float focusDist = 10.f;
  const vec3f origin = lookFrom;
  const vec3f w = normalize(lookFrom - lookAt);
  const vec3f u = normalize(cross(vup, w));
  const vec3f v = cross(w, u);
  const vec3f lower_left_corner
      = origin - half_width * focusDist * u - half_height * focusDist * v - focusDist * w;
  const vec3f horizontal = 2.0f * half_width * focusDist * u;
  const vec3f vertical = 2.0f * half_height * focusDist * v;

  //accumID = 0;

  // ----------- set variables  ----------------------------
  owlRayGenSetGroup(rayGen, "world", world);
  owlRayGenSet3f(rayGen, "camera.org", (const owl3f&)origin);
  owlRayGenSet3f(rayGen, "camera.llc", (const owl3f&)lower_left_corner);
  owlRayGenSet3f(rayGen, "camera.horiz", (const owl3f&)horizontal);
  owlRayGenSet3f(rayGen, "camera.vert", (const owl3f&)vertical);
}
}