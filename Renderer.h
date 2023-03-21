#pragma once
// public owl node-graph API
#include "owl/owl.h"
#include "Camera.h"
#include "CameraManipulator.h"
#include "DeviceCode.h"
#include "GLFWHandler.h"

const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;


namespace deltaVis
{
class Renderer
{
public:
    Renderer();
    ~Renderer();

    void Init();
    void Render();
    void Update();
    void Resize(const vec2i newSize);

    void Shutdown();

    void OnCameraChange();

    Camera camera;
    CameraManipulator* controller;

    OWLRayGen  rayGen  { 0 };
    OWLContext context { 0 };
    OWLGroup   world   { 0 };
    OWLBuffer  accumBuffer { 0 };
    int        accumID     { 0 };
    OWLModule module;
    OWLBuffer frameBuffer;
    OWLBuffer vertexBuffer;
    OWLBuffer indexBuffer;
    OWLGeom trianglesGeom;
    OWLGroup trianglesGroup;
    OWLGeomType trianglesGeomType;
    vec2i fbSize = vec2i(800,600);
};

}