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

Renderer::Renderer()
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
    OWLVarDecl trianglesGeomVars[] = {
        { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
        { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
        { "color",  OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData,color)}
    };
    trianglesGeomType
        = owlGeomTypeCreate(context,
                            OWL_TRIANGLES,
                            sizeof(TrianglesGeomData),
                            trianglesGeomVars,3);
    owlGeomTypeSetClosestHit(trianglesGeomType,0,
                            module,"TriangleMesh");

    
    // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // ------------------------------------------------------------------
  // triangle mesh
  // ------------------------------------------------------------------
  vertexBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,NUM_VERTICES,vertices);
  indexBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,NUM_INDICES,indices);
  frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

  trianglesGeom
    = owlGeomCreate(context,trianglesGeomType);

  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
                          NUM_VERTICES,sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,
                         NUM_INDICES,sizeof(vec3i),0);

  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);
  owlGeomSet3f(trianglesGeom,"color",owl3f{0,1,0});

  // ------------------------------------------------------------------
  // the group/accel for that mesh
  // ------------------------------------------------------------------
  trianglesGroup
    = owlTrianglesGeomGroupCreate(context,1,&trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);
  world
    = owlInstanceGroupCreate(context,1,&trianglesGroup);
  owlGroupBuildAccel(world);


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
  owlMissProgSet3f(missProg,"color0",owl3f{.8f,0.f,0.f});
  owlMissProgSet3f(missProg,"color1",owl3f{.8f,.8f,.8f});


    // -------------------------------------------------------
    // set up ray gen program
    // -------------------------------------------------------
    OWLVarDecl rayGenVars[] = {
        { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
        { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
        { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
        { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
        { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
        { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
        { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
        { /* sentinel to mark end of list */ }
    };

    // ----------- create object  ----------------------------
    rayGen
        = owlRayGenCreate(context,module,"simpleRayGen",
                        sizeof(RayGenData),
                        rayGenVars,-1);

    // ----------- compute variable values  ------------------
    camera.lens.center = lookFrom;
    vec3f camera_d00
        = normalize(lookAt-camera.lens.center);
    float aspect = fbSize.x / float(fbSize.y);
    camera.lens.du
        = cosFovy * aspect * normalize(cross(camera_d00,lookUp));
    camera.lens.dv
        = cosFovy * normalize(cross(camera.lens.du,camera_d00));
    camera_d00 -= 0.5f * camera.lens.du;
    camera_d00 -= 0.5f * camera.lens.dv;

    // ----------- set variables  ----------------------------
    owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
    owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
    owlRayGenSetGroup (rayGen,"world",        world);
    owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera.lens.center);
    owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
    owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera.lens.du);
    owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera.lens.dv);

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

void Renderer::Shutdown()
{
    LOG("destroying devicegroup ...");
    owlContextDestroy(context);
}


void Renderer::UpdateCamera()
{
  vec3f camera_d00
        = normalize(lookAt-camera.lens.center);
  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera.lens.center);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera.lens.du);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera.lens.dv);
}