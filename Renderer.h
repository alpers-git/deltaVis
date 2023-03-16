// public owl node-graph API
#include "owl/owl.h"
#include "DeviceCode.h"

const vec2i fbSize(800,600);
const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;


struct SimpleCamera
{
    inline SimpleCamera() {}
    //SimpleCamera(const SimpleCamera &camera);

    struct {
    vec3f lower_left;
    vec3f horizontal;
    vec3f vertical;
    } screen;
    struct {
    vec3f center;
    vec3f du;
    vec3f dv;
    float radius { 0.f };
    } lens;
};

class Renderer
{
public:
    Renderer();
    ~Renderer();

    void Init();
    void Render();
    void Shutdown();

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