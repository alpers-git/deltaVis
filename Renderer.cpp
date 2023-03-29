#include "Renderer.h"

#define LOG(message)                                          \
  std::cout << OWL_TERMINAL_BLUE;                             \
  std::cout << "#owl.sample(main): " << message << std::endl; \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                       \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
  std::cout << "#owl.sample(main): " << message << std::endl; \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

const int NUM_VERTICES = 8;
vec3f vertices[NUM_VERTICES] =
    {
        {-1.f, -1.f, -1.f},
        {+1.f, -1.f, -1.f},
        {-1.f, +1.f, -1.f},
        {+1.f, +1.f, -1.f},
        {-1.f, -1.f, +1.f},
        {+1.f, -1.f, +1.f},
        {-1.f, +1.f, +1.f},
        {+1.f, +1.f, +1.f}};

const int NUM_INDICES = 12;
vec3i indices[NUM_INDICES] =
    {
        {0, 1, 3}, {2, 3, 0}, {5, 7, 6}, {5, 6, 4}, {0, 4, 5}, {0, 5, 1}, {2, 3, 7}, {2, 7, 6}, {1, 5, 7}, {1, 7, 3}, {4, 0, 2}, {4, 2, 6}};

OWLVarDecl rayGenVars[] = {
    // framebuffer
    {"fbPtr", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, fbPtr)},
    {"fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize)},
    {"world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world)},
    // camera
    {"camera.org", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.origin)},
    {"camera.llc", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.lower_left_corner)},
    {"camera.horiz", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.horizontal)},
    {"camera.vert", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.vertical)},
    // Volume data
    {"volume.elementTLAS", OWL_GROUP, OWL_OFFSETOF(RayGenData, volume.elementTLAS)},
    {/* sentinel to mark end of list */}};
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
    // ##################################################################
    // init owl
    // ##################################################################
    std::string ptx = std::string(deviceCode_ptx);
    ptx[ptx.size() - 1] = 0;
    // create a context on the first device:
    context = owlContextCreate(nullptr, 1);
    module = owlModuleCreate(context, deviceCode_ptx);

    // ##################################################################
    // set miss and raygen program required for SBT
    // ##################################################################

    // -------------------------------------------------------
    // set up miss prog
    // -------------------------------------------------------
    OWLVarDecl missProgVars[] = {
        {"color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color0)},
        {"color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color1)},
        {/* sentinel to mark end of list */}};
    // ----------- create object  ----------------------------
    OWLMissProg missProg = owlMissProgCreate(context, module, "miss", sizeof(MissProgData),
                                             missProgVars, -1);

    // -------------------------------------------------------
    // set up ray gen program
    // -------------------------------------------------------
    rayGen = owlRayGenCreate(context, module, "simpleRayGen",
                             sizeof(RayGenData),
                             rayGenVars, -1);
    // -------------------------------------------------------
    // declare geometry type
    // -------------------------------------------------------
    // Different intersection programs for different element types
    OWLVarDecl unstructuredElementVars[] = {
        {"tetrahedra", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, tetrahedra)},
        {"pyramids", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, pyramids)},
        {"hexahedra", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, hexahedra)},
        {"wedges", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, wedges)},
        {"bytesPerIndex", OWL_UINT, OWL_OFFSETOF(UnstructuredElementData, bytesPerIndex)},
        {"vertices", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, vertices)},
        {"scalars", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, scalars)},
        {"offset", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, offset)},
        {"numTetrahedra", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numTetrahedra)},
        {"numPyramids", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numPyramids)},
        {"numWedges", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numWedges)},
        {"numHexahedra", OWL_ULONG, OWL_OFFSETOF(UnstructuredElementData, numHexahedra)},
        {"maxima", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, maxima)},
        {"bboxes", OWL_BUFPTR, OWL_OFFSETOF(UnstructuredElementData, bboxes)},
        {/* sentinel to mark end of list */}};

    OWLVarDecl triangleVars[] = {
        {"triVertices", OWL_BUFPTR, OWL_OFFSETOF(TriangleData, vertices)},
        {"indices", OWL_BUFPTR, OWL_OFFSETOF(TriangleData, indices)},
        {"color", OWL_FLOAT3, OWL_OFFSETOF(TriangleData, color)},
        {/* sentinel to mark end of list */}};

    // Declare the geometry types
    tetrahedraType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    pyramidType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    wedgeType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    hexahedraType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    triangleType = owlGeomTypeCreate(context, OWL_GEOMETRY_TRIANGLES, sizeof(TriangleData), triangleVars, -1);

    // Set intersection programs
    owlGeomTypeSetIntersectProg(tetrahedraType, /*ray type */ 0, module, "TetrahedraPointQuery");
    // owlGeomTypeSetIntersectProg(pyramidType, /*ray type */ 0, module, "PyramidPointQuery");
    // owlGeomTypeSetIntersectProg(wedgeType, /*ray type */ 0, module, "WedgePointQuery");
    // owlGeomTypeSetIntersectProg(hexahedraType, /*ray type */ 0, module, "HexahedraPointQuery");

    // Set boundary programs
    owlGeomTypeSetBoundsProg(tetrahedraType, module, "TetrahedraBounds");
    // owlGeomTypeSetBoundsProg(pyramidType, module, "PyramidBounds");
    // owlGeomTypeSetBoundsProg(wedgeType, module, "WedgeBounds");
    // owlGeomTypeSetBoundsProg(hexahedraType, module, "HexahedraBounds");

    owlGeomTypeSetClosestHit(triangleType, 0, module, "TriangleClosestHit");

    owlBuildPrograms(context);
    // owlBuildPipeline(context);

    // ##################################################################
    // set up all the *GEOMS* we want to run that code on
    // ##################################################################

    LOG("building geometries ...");
    frameBuffer = owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);
    // ----------- set variables  ----------------------------
    owlMissProgSet3f(missProg, "color0", owl3f{.2f, .2f, .26f});
    owlMissProgSet3f(missProg, "color1", owl3f{.1f, .1f, .16f});
    // ----------- set variables  ----------------------------
    owlRayGenSetBuffer(rayGen, "fbPtr", frameBuffer);
    owlRayGenSet2i(rayGen, "fbSize", (const owl2i &)fbSize);
    owlRayGenSetGroup(rayGen, "world", world);

    // Allocate buffers
    tetrahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->tets.size() * 4, nullptr);
    pyramidsData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->pyrs.size() * 5, nullptr);
    wedgesData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->wedges.size() * 6, nullptr);
    hexahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->hexes.size() * 8, nullptr);
    verticesData = owlDeviceBufferCreate(context, OWL_FLOAT3, umeshPtr->vertices.size(), nullptr);
    scalarData = owlDeviceBufferCreate(context, OWL_FLOAT, umeshPtr->perVertex->values.size(), nullptr);
    // Upload data
    //go over indices of tets and write them in a flat int vector
    std::vector<unsigned int> tetsIndices;
    for (auto &tet : umeshPtr->tets)
    {
      tetsIndices.push_back(tet[0]);
      tetsIndices.push_back(tet[1]);
      tetsIndices.push_back(tet[2]);
      tetsIndices.push_back(tet[3]);
    }
    owlBufferUpload(tetrahedraData, tetsIndices.data());
    owlBufferUpload(pyramidsData, umeshPtr->pyrs.data());
    owlBufferUpload(wedgesData, umeshPtr->wedges.data());
    owlBufferUpload(hexahedraData, umeshPtr->hexes.data());
    owlBufferUpload(verticesData, umeshPtr->vertices.data());
    owlBufferUpload(scalarData, umeshPtr->perVertex->values.data());

    if (umeshPtr->tets.size() > 0)
    {
      OWLGeom tetrahedraGeom = owlGeomCreate(context, tetrahedraType);
      owlGeomSetPrimCount(tetrahedraGeom, umeshPtr->tets.size());
      owlGeomSetBuffer(tetrahedraGeom, "tetrahedra", tetrahedraData);
      owlGeomSetBuffer(tetrahedraGeom, "vertices", verticesData);
      owlGeomSetBuffer(tetrahedraGeom, "scalars", scalarData);
      owlGeomSet1ul(tetrahedraGeom, "offset", 0);
      owlGeomSet1ui(tetrahedraGeom, "bytesPerIndex", 4);
      owlGeomSet1ul(tetrahedraGeom, "numTetrahedra", umeshPtr->tets.size());
      owlGeomSet1ul(tetrahedraGeom, "numPyramids", umeshPtr->pyrs.size());
      owlGeomSet1ul(tetrahedraGeom, "numWedges", umeshPtr->wedges.size());
      owlGeomSet1ul(tetrahedraGeom, "numHexahedra", umeshPtr->hexes.size());
      OWLGroup tetBLAS = owlUserGeomGroupCreate(context, 1, &tetrahedraGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
      owlGroupBuildAccel(tetBLAS);
      elementBLAS.push_back(tetBLAS);
      elementGeom.push_back(tetrahedraGeom);
    }

    if (umeshPtr->pyrs.size() > 0)
    {
      OWLGeom pyramidGeom = owlGeomCreate(context, pyramidType);
      owlGeomSetPrimCount(pyramidGeom, umeshPtr->pyrs.size());
      owlGeomSetBuffer(pyramidGeom, "pyramids", pyramidsData);
      owlGeomSetBuffer(pyramidGeom, "vertices", verticesData);
      owlGeomSetBuffer(pyramidGeom, "scalars", scalarData);
      owlGeomSet1ul(pyramidGeom, "offset", 0);
      owlGeomSet1ui(pyramidGeom, "bytesPerIndex", 4);
      owlGeomSet1ul(pyramidGeom, "numTetrahedra", umeshPtr->tets.size());
      owlGeomSet1ul(pyramidGeom, "numPyramids", umeshPtr->pyrs.size());
      owlGeomSet1ul(pyramidGeom, "numWedges", umeshPtr->wedges.size());
      owlGeomSet1ul(pyramidGeom, "numHexahedra", umeshPtr->hexes.size());
      OWLGroup pyramidBLAS = owlUserGeomGroupCreate(context, 1, &pyramidGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
      owlGroupBuildAccel(pyramidBLAS);
      elementBLAS.push_back(pyramidBLAS);
      elementGeom.push_back(pyramidGeom);
    }

    if (umeshPtr->wedges.size() > 0)
    {
      OWLGeom wedgeGeom = owlGeomCreate(context, wedgeType);
      owlGeomSetPrimCount(wedgeGeom, umeshPtr->wedges.size());
      owlGeomSetBuffer(wedgeGeom, "wedges", wedgesData);
      owlGeomSetBuffer(wedgeGeom, "vertices", verticesData);
      owlGeomSetBuffer(wedgeGeom, "scalars", scalarData);
      owlGeomSet1ul(wedgeGeom, "offset", 0);
      owlGeomSet1ui(wedgeGeom, "bytesPerIndex", 4);
      owlGeomSet1ul(wedgeGeom, "numTetrahedra", umeshPtr->tets.size());
      owlGeomSet1ul(wedgeGeom, "numPyramids", umeshPtr->pyrs.size());
      owlGeomSet1ul(wedgeGeom, "numWedges", umeshPtr->wedges.size());
      owlGeomSet1ul(wedgeGeom, "numHexahedra", umeshPtr->hexes.size());
      OWLGroup wedgeBLAS = owlUserGeomGroupCreate(context, 1, &wedgeGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
      owlGroupBuildAccel(wedgeBLAS);
      elementBLAS.push_back(wedgeBLAS);
      elementGeom.push_back(wedgeGeom);
    }

    if (umeshPtr->hexes.size() > 0)
    {
      OWLGeom hexahedraGeom = owlGeomCreate(context, hexahedraType);
      owlGeomSetPrimCount(hexahedraGeom, umeshPtr->hexes.size());
      owlGeomSetBuffer(hexahedraGeom, "hexahedra", hexahedraData);
      owlGeomSetBuffer(hexahedraGeom, "vertices", verticesData);
      owlGeomSetBuffer(hexahedraGeom, "scalars", scalarData);
      owlGeomSet1ul(hexahedraGeom, "offset", 0);
      owlGeomSet1ui(hexahedraGeom, "bytesPerIndex", 4);
      owlGeomSet1ul(hexahedraGeom, "numTetrahedra", umeshPtr->tets.size());
      owlGeomSet1ul(hexahedraGeom, "numPyramids", umeshPtr->pyrs.size());
      owlGeomSet1ul(hexahedraGeom, "numWedges", umeshPtr->wedges.size());
      owlGeomSet1ul(hexahedraGeom, "numHexahedra", umeshPtr->hexes.size());
      OWLGroup hexBLAS = owlUserGeomGroupCreate(context, 1, &hexahedraGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
      owlGroupBuildAccel(hexBLAS);
      elementBLAS.push_back(hexBLAS);
      elementGeom.push_back(hexahedraGeom);
    }

    elementTLAS = owlInstanceGroupCreate(context, elementBLAS.size(), nullptr, nullptr, nullptr, OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
    for (int i = 0; i < elementBLAS.size(); ++i)
    {
      size_t peak = 0;
      size_t final = 0;
      owlInstanceGroupSetChild(elementTLAS, i, elementBLAS[i]);
      owlGroupGetAccelSize(elementBLAS[i], &final, &peak);
    }
    owlGroupBuildAccel(elementTLAS);
    // owlParamsSetGroup(rayGenVars, "volume.elementTLAS", elementTLAS);
    owlRayGenSetGroup(rayGen, "volume.elementTLAS", elementTLAS);

    size_t peak = 0;
    size_t final = 0;
    owlGroupGetAccelSize(elementTLAS, &final, &peak);

    // ------------------------------------------------------------------
    // Triangle meshes(for surfaces)
    // ------------------------------------------------------------------
    trianglesGeom = owlGeomCreate(context, triangleType);
    indexBuffer = owlDeviceBufferCreate(context, OWL_INT3, NUM_INDICES, indices);
    vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, NUM_VERTICES, vertices);
    owlTrianglesSetIndices(trianglesGeom, indexBuffer, NUM_INDICES, sizeof(vec3i), 0);
    owlTrianglesSetVertices(trianglesGeom, vertexBuffer, NUM_VERTICES, sizeof(vec3f), 0);
    owlGeomSetBuffer(trianglesGeom, "indices", indexBuffer);
    owlGeomSetBuffer(trianglesGeom, "triVertices", vertexBuffer);
    owlGeomSet3f(trianglesGeom, "color", owl3f{0, 1, 0});

    // the group/accel for that mesh (for surfaces)
    trianglesGroup = owlTrianglesGeomGroupCreate(context, 1, &trianglesGeom);
    owlGroupBuildAccel(trianglesGroup);
    world = owlInstanceGroupCreate(context, 1, &trianglesGroup);
    owlGroupBuildAccel(world);
    // owlParamsSetGroup(lp, "trianglesTLAS", trianglesTLAS);

    // go over the vertices of the scene calculate the bounding box and find the center
    auto center = umeshPtr->getBounds().center();
    vec3f eye = vec3f(center.x, center.y, center.z + 2.5f * (umeshPtr->getBounds().upper.z - umeshPtr->getBounds().lower.z));
    camera.setOrientation(eye, vec3f(center.x, center.y, center.z), vec3f(0, 1, 0), 45.0f);
    camera.motionSpeed = 10.f;

    printf("data bounds size = %f %f %f\n", umeshPtr->getBounds().size().x, umeshPtr->getBounds().size().y, umeshPtr->getBounds().size().z);

    // set up camera controller
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
    owlRayGenLaunch2D(rayGen, fbSize.x, fbSize.y);
    // for host pinned mem it doesn't matter which device we query...
    // const uint32_t *fb = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
  }

  void Renderer::Update()
  {
    if (controller->ProcessEvents())
      OnCameraChange();

    auto glfw = GLFWHandler::getInstance();
    if (glfw->getWindowSize() != fbSize)
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
    owlBufferResize(frameBuffer, fbSize.x * fbSize.y);
    owlRayGenSet2i(rayGen, "fbSize", (const owl2i &)fbSize);
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
    const vec3f lower_left_corner = origin - half_width * focusDist * u - half_height * focusDist * v - focusDist * w;
    const vec3f horizontal = 2.0f * half_width * focusDist * u;
    const vec3f vertical = 2.0f * half_height * focusDist * v;

    // accumID = 0;

    // ----------- set variables  ----------------------------
    owlRayGenSetGroup(rayGen, "world", world);
    owlRayGenSet3f(rayGen, "camera.org", (const owl3f &)origin);
    owlRayGenSet3f(rayGen, "camera.llc", (const owl3f &)lower_left_corner);
    owlRayGenSet3f(rayGen, "camera.horiz", (const owl3f &)horizontal);
    owlRayGenSet3f(rayGen, "camera.vert", (const owl3f &)vertical);
  }
}