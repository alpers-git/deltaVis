#include "Renderer.h"
#include "MacrocellBuilder.h"

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
    {nullptr /* sentinel to mark end of list */}};

OWLVarDecl launchParamVars[] = {
    // framebuffer
    {"fbPtr", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, fbPtr)},
    {"fbSize", OWL_INT2, OWL_OFFSETOF(LaunchParams, fbSize)},
    {"shadows", OWL_BOOL, OWL_OFFSETOF(LaunchParams, shadows)},
    {"lightDir", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, lightDir)},
    // accum buffer
    {"accumID", OWL_INT, OWL_OFFSETOF(LaunchParams, accumID)},
    {"accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, accumBuffer)},
    {"frameID", OWL_INT, OWL_OFFSETOF(LaunchParams, frameID)},
    {"triangleTLAS", OWL_GROUP, OWL_OFFSETOF(LaunchParams, triangleTLAS)},
    // camera
    {"camera.org", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.origin)},
    {"camera.llc", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.lower_left_corner)},
    {"camera.horiz", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.horizontal)},
    {"camera.vert", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.vertical)},
    // Volume data
    {"volume.elementTLAS", OWL_GROUP, OWL_OFFSETOF(LaunchParams, volume.elementTLAS)},
    {"volume.macrocellTLAS", OWL_GROUP, OWL_OFFSETOF(LaunchParams, volume.macrocellTLAS)},
    {"volume.macrocellDims", OWL_INT3, OWL_OFFSETOF(LaunchParams, volume.macrocellDims)},
    {"volume.dt", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, volume.dt)},
    {"volume.globalBoundsLo", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, volume.globalBoundsLo)},
    {"volume.globalBoundsHi", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, volume.globalBoundsHi)},
    // transfer function
    {"transferFunction.xf", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, transferFunction.xf)},
    {"transferFunction.volumeDomain", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, transferFunction.volumeDomain)},
    {"transferFunction.opacityScale", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, transferFunction.opacityScale)},
    //{"volume.mecrocells"}
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
    owlContextSetRayTypeCount(context, 2);
    owlContextSetNumPayloadValues(context, 32);

    // ##################################################################
    // set miss and raygen program required for SBT
    // ##################################################################

    // -------------------------------------------------------
    // set up ray gen program
    // -------------------------------------------------------
    rayGen = owlRayGenCreate(context, module, "simpleRayGen",
                             sizeof(RayGenData),
                             rayGenVars, -1);
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

    lp = owlParamsCreate(context, sizeof(LaunchParams), launchParamVars, -1);
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

    OWLVarDecl macrocellVars[] = {
        {"bboxes", OWL_BUFPTR, OWL_OFFSETOF(MacrocellData, bboxes)},
        {"maxima", OWL_BUFPTR, OWL_OFFSETOF(MacrocellData, maxima)},
        {/* sentinel to mark end of list */}};

    // Declare the geometry types
    macrocellType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(MacrocellData), macrocellVars, -1);
    tetrahedraType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    pyramidType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    wedgeType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    hexahedraType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(UnstructuredElementData), unstructuredElementVars, -1);
    triangleType = owlGeomTypeCreate(context, OWL_GEOMETRY_TRIANGLES, sizeof(TriangleData), triangleVars, -1);

    // Set intersection programs
    owlGeomTypeSetIntersectProg(macrocellType, /*ray type */ 1, module, "MacrocellIntersection");
    owlGeomTypeSetIntersectProg(macrocellType, /*ray type */ 0, module, "VolumeIntersection");
    owlGeomTypeSetIntersectProg(tetrahedraType, /*ray type */ 0, module, "TetrahedraPointQuery");
    owlGeomTypeSetIntersectProg(pyramidType, /*ray type */ 0, module, "PyramidPointQuery");
    owlGeomTypeSetIntersectProg(wedgeType, /*ray type */ 0, module, "WedgePointQuery");
    owlGeomTypeSetIntersectProg(hexahedraType, /*ray type */ 0, module, "HexahedraPointQuery");

    // Set boundary programs
    owlGeomTypeSetBoundsProg(tetrahedraType, module, "TetrahedraBounds");
    owlGeomTypeSetBoundsProg(pyramidType, module, "PyramidBounds");
    owlGeomTypeSetBoundsProg(wedgeType, module, "WedgeBounds");
    owlGeomTypeSetBoundsProg(hexahedraType, module, "HexahedraBounds");
    owlGeomTypeSetBoundsProg(macrocellType, module, "MacrocellBounds");

    owlGeomTypeSetClosestHit(triangleType, 0, module, "TriangleClosestHit");
    //owlGeomTypeSetClosestHit(macrocellType, /*ray type */ 0, module,"DeltaTracking");
    owlGeomTypeSetClosestHit(macrocellType, /*ray type */ 0, module, "AdaptiveDeltaTracking");

    owlBuildPrograms(context);
    // owlBuildPipeline(context);

    frameBuffer = owlHostPinnedBufferCreate(context, OWL_INT, fbSize.x * fbSize.y);
    if (!accumBuffer)
      accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(accumBuffer, fbSize.x * fbSize.y);
    owlParamsSetBuffer(lp, "accumBuffer", accumBuffer);
    accumID = 0;
    frameID = 0;
    owlParamsSet1i(lp, "accumID", accumID);
    owlParamsSet1i(lp, "frameID", frameID);
    // ##################################################################
    // set up all the *GEOMS* we want to run that code on
    // ##################################################################

    LOG("building geometries ...");

    // ----------- set variables  ----------------------------
    owlMissProgSet3f(missProg, "color0", owl3f{.2f, .2f, .26f});
    owlMissProgSet3f(missProg, "color1", owl3f{.1f, .1f, .16f});
    // ----------- set variables  ----------------------------
    owlParamsSetBuffer(lp, "fbPtr", frameBuffer);
    owlParamsSet2i(lp, "fbSize", (const owl2i &)fbSize);
    owlParamsSetGroup(lp, "triangleTLAS", triangleTLAS);

    // Allocate buffers
    tetrahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->tets.size() * 4, nullptr);
    pyramidsData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->pyrs.size() * 5, nullptr);
    wedgesData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->wedges.size() * 6, nullptr);
    hexahedraData = owlDeviceBufferCreate(context, OWL_INT, umeshPtr->hexes.size() * 8, nullptr);
    verticesData = owlDeviceBufferCreate(context, OWL_FLOAT3, umeshPtr->vertices.size(), nullptr);
    scalarData = owlDeviceBufferCreate(context, OWL_FLOAT, umeshPtr->perVertex->values.size(), nullptr);

    // Upload data
    owlBufferUpload(tetrahedraData, umeshPtr->tets.data());
    owlBufferUpload(pyramidsData, umeshPtr->pyrs.data());
    owlBufferUpload(wedgesData, umeshPtr->wedges.data());
    owlBufferUpload(hexahedraData, umeshPtr->hexes.data());
    owlBufferUpload(verticesData, umeshPtr->vertices.data());
    owlBufferUpload(scalarData, umeshPtr->perVertex->values.data());

    int numMacrocells = macrocellDims.x * macrocellDims.y * macrocellDims.z;
    box4f *grid = BuildMacrocellGrid(macrocellDims, umeshPtr->vertices.data(),
                                     umeshPtr->perVertex->values.data(), umeshPtr->vertices.size());
    for (int i = 0; i < numMacrocells; i++)
      std::cout << grid[i].lower << " " << grid[i].upper << std::endl;
    gridBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, numMacrocells * 2, nullptr);
    owlBufferUpload(gridBuffer, grid);
    majorantBuffer = owlDeviceBufferCreate(context, OWL_FLOAT, numMacrocells, nullptr);
    OWLGeom userGeom = owlGeomCreate(context, macrocellType);
    owlGeomSetPrimCount(userGeom, numMacrocells);
    // owlGeomSet1i(userGeom, "offset", 0);
    owlGeomSetBuffer(userGeom, "maxima", majorantBuffer);
    owlGeomSetBuffer(userGeom, "bboxes", gridBuffer);

    auto macrocellBLAS = owlUserGeomGroupCreate(context, 1, &userGeom, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
    owlGroupBuildAccel(macrocellBLAS);
    macrocellTLAS = owlInstanceGroupCreate(context, 1, &macrocellBLAS,
                                           nullptr, nullptr, OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
    owlGroupBuildAccel(macrocellTLAS);
    owlParamsSetGroup(lp, "volume.macrocellTLAS", macrocellTLAS);

    owlParamsSet3i(lp, "volume.macrocellDims", (const owl3i &)macrocellDims);

    // delete[] grid;
    cudaDeviceSynchronize();

    if (umeshPtr->tets.size() > 0)
    {
      OWLGeom tetrahedraGeom = owlGeomCreate(context, tetrahedraType);
      owlGeomSetPrimCount(tetrahedraGeom, umeshPtr->tets.size() * 4);
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
    // owlParamsSetGroup(lp, "volume.elementTLAS", elementTLAS);
    owlParamsSetGroup(lp, "volume.elementTLAS", elementTLAS);

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
    triangleTLAS = owlInstanceGroupCreate(context, 1, &trianglesGroup);
    owlGroupBuildAccel(triangleTLAS);

    // go over the vertices of the scene calculate the bounding box and find the center
    auto center = umeshPtr->getBounds().center();
    vec3f eye = vec3f(center.x, center.y, center.z + 2.5f * (umeshPtr->getBounds().upper.z - umeshPtr->getBounds().lower.z));
    camera.setOrientation(eye, vec3f(center.x, center.y, center.z), vec3f(0, 1, 0), 45.0f);

    // set up camera controller
    controller = new CameraManipulator(&camera);
    OnCameraChange();

    // transfer function
    SetOpacityScale(1.0f);
    volDomain = interval<float>({umeshPtr->getBounds4f().lower.w, umeshPtr->getBounds4f().upper.w});
    owlParamsSet4f(lp, "volume.globalBoundsLo",
                   owl4f{umeshPtr->getBounds4f().lower.x, umeshPtr->getBounds4f().lower.y,
                         umeshPtr->getBounds4f().lower.z, umeshPtr->getBounds4f().lower.w});
    owlParamsSet4f(lp, "volume.globalBoundsHi",
                   owl4f{umeshPtr->getBounds4f().upper.x, umeshPtr->getBounds4f().upper.y,
                         umeshPtr->getBounds4f().upper.z, umeshPtr->getBounds4f().upper.w});
    printf("volDomain: %f %f\n", volDomain.lower, volDomain.upper);
    owlParamsSet2f(lp, "transferFunction.volumeDomain", owl2f{volDomain.lower, volDomain.upper});

    double avgBbox = 0.0f;
    double counter = 0;
    for(auto tet : umeshPtr->tets)
    {
      box3f bbox;
      bbox.extend({umeshPtr->vertices[tet[0]].x,
        umeshPtr->vertices[tet[0]].y,
        umeshPtr->vertices[tet[0]].z});
      bbox.extend({umeshPtr->vertices[tet[1]].x,
        umeshPtr->vertices[tet[1]].y,
        umeshPtr->vertices[tet[1]].z});
      bbox.extend({umeshPtr->vertices[tet[2]].x,
        umeshPtr->vertices[tet[2]].y,
        umeshPtr->vertices[tet[2]].z});
      bbox.extend({umeshPtr->vertices[tet[3]].x,
        umeshPtr->vertices[tet[3]].y,
        umeshPtr->vertices[tet[3]].z});
      avgBbox += owl::common::length(bbox.span());
      counter+=1.f;
    }
    for(auto pyr : umeshPtr->pyrs)
    {
      box3f bbox;
      bbox.extend({umeshPtr->vertices[pyr[0]].x,
        umeshPtr->vertices[pyr[0]].y,
        umeshPtr->vertices[pyr[0]].z});
      bbox.extend({umeshPtr->vertices[pyr[1]].x,
        umeshPtr->vertices[pyr[1]].y,
        umeshPtr->vertices[pyr[1]].z});
      bbox.extend({umeshPtr->vertices[pyr[2]].x,
        umeshPtr->vertices[pyr[2]].y,
        umeshPtr->vertices[pyr[2]].z});
      bbox.extend({umeshPtr->vertices[pyr[3]].x,
        umeshPtr->vertices[pyr[3]].y,
        umeshPtr->vertices[pyr[3]].z});
      bbox.extend({umeshPtr->vertices[pyr[4]].x,
        umeshPtr->vertices[pyr[4]].y,
        umeshPtr->vertices[pyr[4]].z});
      avgBbox += owl::common::length(bbox.span());
      counter+=1.f;
    }
    for(auto wed : umeshPtr->wedges)
    {
      box3f bbox;
      bbox.extend({umeshPtr->vertices[wed[0]].x,
        umeshPtr->vertices[wed[0]].y,
        umeshPtr->vertices[wed[0]].z});
      bbox.extend({umeshPtr->vertices[wed[1]].x,
        umeshPtr->vertices[wed[1]].y,
        umeshPtr->vertices[wed[1]].z});
      bbox.extend({umeshPtr->vertices[wed[2]].x,
        umeshPtr->vertices[wed[2]].y,
        umeshPtr->vertices[wed[2]].z});
      bbox.extend({umeshPtr->vertices[wed[3]].x,
        umeshPtr->vertices[wed[3]].y,
        umeshPtr->vertices[wed[3]].z});
      bbox.extend({umeshPtr->vertices[wed[4]].x,
        umeshPtr->vertices[wed[4]].y,
        umeshPtr->vertices[wed[4]].z});
      bbox.extend({umeshPtr->vertices[wed[5]].x,
        umeshPtr->vertices[wed[5]].y,
        umeshPtr->vertices[wed[5]].z});
      avgBbox += owl::common::length(bbox.span());
      counter+=1.f;
    }
    for(auto hex : umeshPtr->hexes)
    {
      box3f bbox;
      bbox.extend({umeshPtr->vertices[hex[0]].x,
        umeshPtr->vertices[hex[0]].y,
        umeshPtr->vertices[hex[0]].z});
      bbox.extend({umeshPtr->vertices[hex[1]].x,
        umeshPtr->vertices[hex[1]].y,
        umeshPtr->vertices[hex[1]].z});
      bbox.extend({umeshPtr->vertices[hex[2]].x,
        umeshPtr->vertices[hex[2]].y,
        umeshPtr->vertices[hex[2]].z});
      bbox.extend({umeshPtr->vertices[hex[3]].x,
        umeshPtr->vertices[hex[3]].y,
        umeshPtr->vertices[hex[3]].z});
      bbox.extend({umeshPtr->vertices[hex[4]].x,
        umeshPtr->vertices[hex[4]].y,
        umeshPtr->vertices[hex[4]].z});
      bbox.extend({umeshPtr->vertices[hex[5]].x,
        umeshPtr->vertices[hex[5]].y,
        umeshPtr->vertices[hex[5]].z});
      bbox.extend({umeshPtr->vertices[hex[6]].x,
        umeshPtr->vertices[hex[6]].y,
        umeshPtr->vertices[hex[6]].z});
      bbox.extend({umeshPtr->vertices[hex[7]].x,
        umeshPtr->vertices[hex[7]].y,
        umeshPtr->vertices[hex[7]].z});
      avgBbox += owl::common::length(bbox.span());
      counter+=1.f;
    }

    avgBbox /= counter;
    dt = avgBbox;

    printf("avg bbox size: %f\n", avgBbox);
    
    // ##################################################################
    // build *SBT* required to trace the groups
    // ##################################################################
    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
    delete[] grid;
  }

  void Renderer::Render(bool headless)
  {
    owlBuildSBT(context);
    // get time start
    auto start = std::chrono::high_resolution_clock::now();
    owlLaunch2D(rayGen, fbSize.x, fbSize.y, lp);
    // get time end
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    lastFrameTime = elapsed.count();
    avgFrameTime = 0.75 * avgFrameTime + 0.25 * elapsed.count();

    accumID++;
    frameID++;
    owlParamsSet1i(lp, "accumID", accumID);
    owlParamsSet1i(lp, "frameID", frameID);
    // for host pinned mem it doesn't matter which device we query...
    // const uint32_t *fb = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
    if(headless)
      printf("frame %i, time %f, avg time %f, fps %f \n", frameID, lastFrameTime,
             avgFrameTime, 1.f / avgFrameTime);
  }

  void Renderer::Update(bool headless)
  {
    if (controller->ProcessEvents())
      OnCameraChange();
    owlParamsSet1f(lp, "volume.dt", dt);
    owlParamsSet1b(lp, "shadows", shadows);

    auto glfw = GLFWHandler::getInstance();
    if (!headless && glfw->getWindowSize() != fbSize)
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
    owlParamsSet2i(lp, "fbSize", (const owl2i &)fbSize);
    if (!accumBuffer)
      accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(accumBuffer, fbSize.x * fbSize.y);
    owlParamsSetBuffer(lp, "accumBuffer", accumBuffer);
    OnCameraChange();
  }

  void Renderer::SetOpacityScale(float scale)
  {
    owlParamsSet1f(lp, "transferFunction.opacityScale", scale);
    accumID = 0;
    RecalculateDensityRanges();
  }

  void Renderer::SetColorMap(const std::vector<vec4f> &newCM)
  {
    std::vector<vec4f> CM = newCM;
    for (uint32_t i = 0; i < CM.size(); ++i)
    {
      CM[i].w = powf(CM[i].w, 3.f);
    }

    this->colorMap = CM;
    if (!colorMapBuffer)
      colorMapBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4,
                                             CM.size(), nullptr);
    owlBufferUpload(colorMapBuffer, CM.data());

    if (colorMapTexture != 0)
    {
      (cudaDestroyTextureObject(colorMapTexture));
      colorMapTexture = 0;
    }

    cudaResourceDesc res_desc = {};
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

    // cudaArray_t   voxelArray;
    if (colorMapArray == 0)
    {
      (cudaMallocArray(&colorMapArray,
                       &channel_desc,
                       CM.size(), 1));
    }

    int pitch = CM.size() * sizeof(CM[0]);
    (cudaMemcpy2DToArray(colorMapArray,
                         /* offset */ 0, 0,
                         CM.data(),
                         pitch, pitch, 1,
                         cudaMemcpyHostToDevice));

    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = colorMapArray;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 0.0f;
    tex_desc.borderColor[1] = 0.0f;
    tex_desc.borderColor[2] = 0.0f;
    tex_desc.borderColor[3] = 0.0f;
    tex_desc.sRGB = 0;
    (cudaCreateTextureObject(&colorMapTexture, &res_desc, &tex_desc,
                             nullptr));

    // OWLTexture xfTexture
    //   = owlTexture2DCreate(owl,OWL_TEXEL_FORMAT_RGBA32F,
    //                        colorMap.size(),1,
    //                        colorMap.data());
    owlParamsSetRaw(lp, "transferFunction.xf", &colorMapTexture);
    accumID = 0;
    owlParamsSet1i(lp, "accumID", accumID);
    RecalculateDensityRanges();
  }

  void Renderer::SetLightDir(const vec3f &dir)
  {
    lightDir = dir;
    owlParamsSet3f(lp, "lightDir", (const owl3f &)dir);
    accumID = 0;
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
    camera.motionSpeed = umesh::length(umeshPtr->getBounds().size()) / 50.f;

    accumID = 0;

    // ----------- set variables  ----------------------------
    owlParamsSetGroup(lp, "triangleTLAS", triangleTLAS);
    owlParamsSet3f(lp, "camera.org", (const owl3f &)origin);
    owlParamsSet3f(lp, "camera.llc", (const owl3f &)lower_left_corner);
    owlParamsSet3f(lp, "camera.horiz", (const owl3f &)horizontal);
    owlParamsSet3f(lp, "camera.vert", (const owl3f &)vertical);
    owlParamsSet1i(lp, "accumID", accumID);
  }
}