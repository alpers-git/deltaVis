#pragma once
// public owl node-graph API
#include "owl/owl.h"
#include "owl/owl_host.h"
#include "Camera.h"
#include "CameraManipulator.h"
#include "DeviceCode.h"
#include "GLFWHandler.h"

#include <umesh/io/UMesh.h>


const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;


namespace deltaVis
{
class Renderer
{
public:
    Renderer(std::shared_ptr<umesh::UMesh> umesh);
    ~Renderer();

    void Init();
    void Render(bool headless = false);
    void Update(bool headless = false);
    void Resize(const vec2i newSize);
    void SetOpacityScale(float scale);
    void SetColorMap(const std::vector<owl::vec4f> &newXF);
    void SetLightDir(const owl::vec3f &newDir);
    void RecalculateDensityRanges();

    void Shutdown();

    void OnCameraChange();

    std::shared_ptr<umesh::UMesh> umeshPtr;

    Camera camera;
    CameraManipulator* controller;

    OWLRayGen  rayGen  { 0 };
    OWLParams lp       { 0 };
    OWLContext context { 0 };
    OWLGroup   triangleTLAS   { 0 };
    OWLBuffer  accumBuffer { 0 };
    int        accumID     { 0 };
    int        frameID     { 0 };
    OWLModule module;

    float dt = 0.5f;
    bool shadows = false;
    vec3f lightDir = normalize(vec3f(0.f,-1.f,0.f));
    double avgFrameTime = 0.0;
    double lastFrameTime = 0.0;

    std::vector<OWLGeom> elementGeom;
    std::vector<OWLGroup> elementBLAS;
    OWLGroup elementTLAS;
    //OWLGroup rootMacrocellBLAS;
    OWLGroup macrocellTLAS;


    OWLGeom trianglesGeom;
    OWLGroup trianglesGroup;

    OWLGeomType macrocellType;
    OWLGeomType tetrahedraType;
    OWLGeomType pyramidType;
    OWLGeomType wedgeType;
    OWLGeomType hexahedraType;

    OWLGeomType triangleType;

    OWLBuffer  tetrahedraData;
    OWLBuffer  pyramidsData;
    OWLBuffer  hexahedraData;
    OWLBuffer  wedgesData;
    OWLBuffer  verticesData;
    OWLBuffer  scalarData;
    OWLBuffer gridBuffer;
    OWLBuffer majorantBuffer;

    OWLBuffer vertexBuffer;
    OWLBuffer indexBuffer;
    OWLBuffer frameBuffer;

    vec2i fbSize = vec2i(800,600);
    vec3i macrocellDims = {4,4,4};


    OWLBuffer colorMapBuffer { 0 };
    cudaArray_t colorMapArray { 0 };
    cudaTextureObject_t colorMapTexture { 0 };

    interval<float> volDomain;
    float opacityScale = 1.f;
    std::vector<vec4f> colorMap;
};

}