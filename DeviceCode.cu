#include "DeviceCode.h"
#include <optix_device.h>

using namespace deltaVis;
using namespace owl;

#define DEBUG 0
// create a debug function macro that gets called only for center pixel
inline __device__ bool dbg()
{
#if DEBUG
  return false;
#else
  auto pixelID = vec2i(owl::getLaunchIndex()[0], owl::getLaunchIndex()[1]);
  return (owl::getProgramData<RayGenData>().fbSize.x / 2 == pixelID.x) &&
         (owl::getProgramData<RayGenData>().fbSize.y / 2 == pixelID.y);
#define ACTIVATE_CROSSHAIRS
#endif
}

inline __device__
    float4
    transferFunction(float f, float2 volumeDomain)
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  if (f < volumeDomain.x ||
      f > volumeDomain.y)
  {
    // gradient = 0.f;
    return make_float4(1.f, 0.f, 1.f, 0.5f);
  }
  float remapped = (f - volumeDomain.x) /
                   (volumeDomain.y - volumeDomain.x);

  float4 xf = tex2D<float4>(self.transferFunction.xf, remapped, 0.5f);
  xf.w *= self.transferFunction.opacityScale;

  return xf;
}

inline __device__ void generateRay(const vec2f screen, owl::Ray &ray)
{
  auto &self = owl::getProgramData<RayGenData>();
  ray.origin = self.camera.origin;
  vec3f direction = self.camera.lower_left_corner +
                    screen.u * self.camera.horizontal +
                    screen.v * self.camera.vertical - ray.origin;
  direction = normalize(direction);
  if (fabs(direction.x) < 1e-5f)
    direction.x = 1e-5f;
  if (fabs(direction.y) < 1e-5f)
    direction.y = 1e-5f;
  if (fabs(direction.z) < 1e-5f)
    direction.z = 1e-5f;
  ray.direction = normalize(direction);
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)
()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  // if (pixelID == owl::vec2i(0)) {
  //   printf("%sHello OptiX From your First RayGen Program%s\n",
  //          OWL_TERMINAL_CYAN,
  //          OWL_TERMINAL_DEFAULT);
  // }
  int seed = owl::getLaunchDims().x * owl::getLaunchDims().y * self.frameID;
  owl::common::LCG<4> random(threadIdx.x + seed, threadIdx.y + seed);
  const vec2f screen = (vec2f(pixelID) + vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  generateRay(screen, ray);
  // ray.origin = ray.origin + random() * self.camera.horizontal +
  //              random() * self.camera.vertical;

  RayPayload prd;
  float count = 0;
  prd.missed = true;
  prd.rgba = vec4f(0, 0, 0, 0);
  // prd.dataValue = 0;
  prd.debug = dbg();
  owl::traceRay(/*accel to trace against*/ self.volume.macrocellTLAS,
                /*the ray to trace*/ ray,
                /*prd*/ prd);
  if (!prd.missed)
    count += 0.1f;
  // map prd.dataValue to color
  float4 tfColor = transferFunction(prd.dataValue, self.transferFunction.volumeDomain);
  vec4f color = prd.missed ? prd.rgba : vec4f(tfColor.x, tfColor.y, tfColor.z, tfColor.w);
  // vec3f color = vec3f(prd.rgba.x, prd.rgba.y, prd.rgba.z);

  const int fbOfs = pixelID.x + self.fbSize.x * pixelID.y;
  // self.fbPtr[fbOfs] = owl::make_rgba(color);
  vec4f oldColor = self.accumBuffer[fbOfs];
  vec4f newColor = (vec4f(color) + float(self.accumID) * oldColor) / float(self.accumID + 1);
  self.fbPtr[fbOfs] = make_rgba(vec4f(newColor));
  self.accumBuffer[fbOfs] = vec4f(newColor);

#ifdef ACTIVATE_CROSSHAIRS
  if (pixelID.x == self.fbSize.x / 2 || pixelID.y == self.fbSize.y / 2 ||
      pixelID.x == self.fbSize.x / 2 + 1 || pixelID.y == self.fbSize.y / 2 + 1 ||
      pixelID.x == self.fbSize.x / 2 - 1 || pixelID.y == self.fbSize.y / 2 - 1)
    self.fbPtr[fbOfs] = owl::make_rgba(color * 0.33f);
#endif
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleClosestHit)
()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TriangleData &self = owl::getProgramData<TriangleData>();

  // compute normal:
  const int primID = optixGetPrimitiveIndex();
  const vec3i index = self.indices[primID];
  const vec3f &A = self.vertices[index.x];
  const vec3f &B = self.vertices[index.y];
  const vec3f &C = self.vertices[index.z];
  const vec3f Ng = normalize(cross(B - A, C - A));

  const vec3f rayDir = optixGetWorldRayDirection();
  prd = (.2f + .8f * fabs(dot(rayDir, Ng))) * self.color;
}

OPTIX_CLOSEST_HIT_PROGRAM(DeltaTracking)
()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  RayPayload &prd = owl::getPRD<RayPayload>();
  // if (prd.debug)
  //   printf("hit delta cell\n");
  // prd.missed = false;
  // auto &lp = optixLaunchParams;
  vec3f origin = vec3f(optixGetWorldRayOrigin());
  vec3f direction = vec3f(optixGetWorldRayDirection());

  // auto sampler = Sampler(prd.debug);
  RayPayload samplePrd;
  owl::Ray sampleRay;
  sampleRay.origin = origin;
  sampleRay.direction = direction;
  owl::traceRay(self.volume.elementTLAS, sampleRay, samplePrd);
  if(prd.debug)
    printf("samplePrd.dataValue: %f\n", samplePrd.dataValue);

  //   float majorantExtinction = self.maxima[0];
  //   if (majorantExtinction == 0.f)
  //     return;

  //   const interval<float> xfDomain = lp.transferFunc.domain;
  //   const interval<float> volDomain = lp.volume.domain;
  //   float unit = lp.volume.dt;
  //   float t = prd.t0;
  //   // while (true) {
  //   for (int i = 0; i < MAX_VOLUME_DEPTH; ++i)
  //   {
  //     // Sample a distance
  //     t = t - (log(1.0f - prd.random()) / majorantExtinction) * unit;

  //     // A boundary has been hit
  //     if (t >= prd.t1)
  //       break;

  //     // Update current position
  //     vec3f x = origin + t * direction;

  //     // Sample heterogeneous media
  //     float dataValue = sampler({x.x, x.y, x.z});
  //     prd.samples++;
  //     float4 xf = make_float4(0.f, 0.f, 0.f, 0.f);
  //     if (dataValue != sampler.background())
  //     {
  // #ifdef RENDER_LEVEL
  //       // if (prd.debug) printf("%f\n", dataValue);
  //       float remapped2 = ((dataValue / 12.f) - xfDomain.lower) / (xfDomain.upper - xfDomain.lower);
  //       xf = tex2D<float4>(lp.transferFunc.texture, remapped2, 0.5f);
  //       xf.w *= lp.transferFunc.opacityScale;
  // #else
  //       float remapped1 = (dataValue - volDomain.lower) / (volDomain.upper - volDomain.lower);
  //       float remapped2 = (remapped1 - xfDomain.lower) / (xfDomain.upper - xfDomain.lower);
  //       xf = tex2D<float4>(lp.transferFunc.texture, remapped2, 0.5f);
  //       xf.w *= lp.transferFunc.opacityScale;
  // #endif
  //     }

  //     // Check if an emission occurred
  //     if (prd.random() < xf.w / (majorantExtinction))
  //     {
  //       prd.tHit = min(prd.tHit, t);
  //       prd.rgba = vec4f(vec3f(xf), 1.f);
  //       break;
  //     }
  //   }
}

OPTIX_MISS_PROGRAM(miss)
()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();

  RayPayload &prd = owl::getPRD<RayPayload>();
  int pattern = (pixelID.x / 18) ^ (pixelID.y / 18);
  prd.rgba = (pattern & 1) ? vec4f(self.color1, 1) : vec4f(self.color0, 1);
  prd.missed = true;
}

// ------------------------------------------------------------------
// Bounds programs for volume elements
// ------------------------------------------------------------------

OPTIX_BOUNDS_PROGRAM(MacrocellBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID)
{
  const MacrocellData &self = *(const MacrocellData *)geomData;
  // if (self.maxima[primID] <= 0.f) {
  //    primBounds = box3f(); // empty box
  //  }
  //  else
  {
    primBounds = box3f();
    primBounds = primBounds.including(vec3f(self.bboxes[(primID * 2 + 0)].x,
                                            self.bboxes[(primID * 2 + 0)].y,
                                            self.bboxes[(primID * 2 + 0)].z) -
                                      vec3f(0.001f, 0.001f, 0.001f));
    primBounds = primBounds.including(vec3f(self.bboxes[(primID * 2 + 1)].x,
                                            self.bboxes[(primID * 2 + 1)].y,
                                            self.bboxes[(primID * 2 + 1)].z) +
                                      vec3f(0.001f, 0.001f, 0.001f));
    // primBounds.lower.x = self.bboxes[(primID * 2 + 0)].x;
    // primBounds.lower.y = self.bboxes[(primID * 2 + 0)].y;
    // primBounds.lower.z = self.bboxes[(primID * 2 + 0)].z;
    // primBounds.upper.x = self.bboxes[(primID * 2 + 1)].x;
    // primBounds.upper.y = self.bboxes[(primID * 2 + 1)].y;
    // primBounds.upper.z = self.bboxes[(primID * 2 + 1)].z;
  }
}

OPTIX_BOUNDS_PROGRAM(TetrahedraBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID)
{
  const UnstructuredElementData &self = *(const UnstructuredElementData *)geomData;
  primBounds = box3f();
  unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;
  if (ID >= self.numTetrahedra)
    return;

  unsigned int *tets = (unsigned int *)self.tetrahedra;
  uint64_t i0 = tets[ID * 4 + 0];
  uint64_t i1 = tets[ID * 4 + 1];
  uint64_t i2 = tets[ID * 4 + 2];
  uint64_t i3 = tets[ID * 4 + 3];

  vec3f P0 = self.vertices[i0];
  vec3f P1 = self.vertices[i1];
  vec3f P2 = self.vertices[i2];
  vec3f P3 = self.vertices[i3];

  primBounds = primBounds.including(P0)
                   .including(P1)
                   .including(P2)
                   .including(P3);
  // printf("IDs: %d %d %d %d\n", i0, i1, i2, i3);
}

// OPTIX_BOUNDS_PROGRAM(PyramidBounds)(
//   const void  *geomData,
//   owl::common::box3f &primBounds,
//   const int    primID,
//   const int    key)

// {
//   const UnstructuredElementData &self = *(const UnstructuredElementData*)geomData;
//   primBounds = box3f();
//   for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
//     uint32_t ID = (uint32_t(primID) + self.offset) * ELEMENTS_PER_BOX + i;
//     if (ID >= self.numPyramids) return;

//     uint64_t i0, i1, i2, i3, i4;
//     if (self.bytesPerIndex == 1) {
//       uint8_t* pyrs = (uint8_t*)self.pyramids;
//       i0 = pyrs[ID * 5 + 0];
//       i1 = pyrs[ID * 5 + 1];
//       i2 = pyrs[ID * 5 + 2];
//       i3 = pyrs[ID * 5 + 3];
//       i4 = pyrs[ID * 5 + 4];
//     } else if (self.bytesPerIndex == 2) {
//       uint16_t* pyrs = (uint16_t*)self.pyramids;
//       i0 = pyrs[ID * 5 + 0];
//       i1 = pyrs[ID * 5 + 1];
//       i2 = pyrs[ID * 5 + 2];
//       i3 = pyrs[ID * 5 + 3];
//       i4 = pyrs[ID * 5 + 4];
//     } else {
//       uint32_t* pyrs = (uint32_t*)self.pyramids;
//       i0 = pyrs[ID * 5 + 0];
//       i1 = pyrs[ID * 5 + 1];
//       i2 = pyrs[ID * 5 + 2];
//       i3 = pyrs[ID * 5 + 3];
//       i4 = pyrs[ID * 5 + 4];
//     }

//     vec3f P0 = self.vertices[i0];
//     vec3f P1 = self.vertices[i1];
//     vec3f P2 = self.vertices[i2];
//     vec3f P3 = self.vertices[i3];
//     vec3f P4 = self.vertices[i4];
//     primBounds = primBounds
//       .including(P0)
//       .including(P1)
//       .including(P2)
//       .including(P3)
//       .including(P4);
//   }
// }

// OPTIX_BOUNDS_PROGRAM(WedgeBounds)(
//   const void  *geomData,
//   owl::common::box3f &primBounds,
//   const int    primID,
//   const int    key)
// {
//   const UnstructuredElementData &self = *(const UnstructuredElementData*)geomData;
//   primBounds = box3f();
//   for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
//     uint32_t ID = (uint32_t(primID) + self.offset) * ELEMENTS_PER_BOX + i;
//     if (ID >= self.numWedges) return;

//     uint64_t i0, i1, i2, i3, i4, i5;
//     if (self.bytesPerIndex == 1) {
//       uint8_t* wed = (uint8_t*)self.wedges;
//       i0 = wed[ID * 6 + 0];
//       i1 = wed[ID * 6 + 1];
//       i2 = wed[ID * 6 + 2];
//       i3 = wed[ID * 6 + 3];
//       i4 = wed[ID * 6 + 4];
//       i5 = wed[ID * 6 + 5];
//     } else if (self.bytesPerIndex == 2) {
//       uint16_t* wed = (uint16_t*)self.wedges;
//       i0 = wed[ID * 6 + 0];
//       i1 = wed[ID * 6 + 1];
//       i2 = wed[ID * 6 + 2];
//       i3 = wed[ID * 6 + 3];
//       i4 = wed[ID * 6 + 4];
//       i5 = wed[ID * 6 + 5];
//     } else {
//       uint32_t* wed = (uint32_t*)self.wedges;
//       i0 = wed[ID * 6 + 0];
//       i1 = wed[ID * 6 + 1];
//       i2 = wed[ID * 6 + 2];
//       i3 = wed[ID * 6 + 3];
//       i4 = wed[ID * 6 + 4];
//       i5 = wed[ID * 6 + 5];
//     }

//     vec3f P0 = self.vertices[i0];
//     vec3f P1 = self.vertices[i1];
//     vec3f P2 = self.vertices[i2];
//     vec3f P3 = self.vertices[i3];
//     vec3f P4 = self.vertices[i4];
//     vec3f P5 = self.vertices[i5];
//     primBounds = primBounds
//       .including(P0)
//       .including(P1)
//       .including(P2)
//       .including(P3)
//       .including(P4)
//       .including(P5);
//   }
// }

// OPTIX_BOUNDS_PROGRAM(HexahedraBounds)(
//   const void  *geomData,
//   owl::common::box3f &primBounds,
//   const int    primID,
//   const int    key)
// {
//   const UnstructuredElementData &self = *(const UnstructuredElementData*)geomData;
//   primBounds = box3f();
//   for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
//     uint32_t ID = (uint32_t(primID) + self.offset) * ELEMENTS_PER_BOX + i;
//     if (ID >= self.numHexahedra) return;

//     uint64_t i0, i1, i2, i3, i4, i5, i6, i7;
//     if (self.bytesPerIndex == 1) {
//       uint8_t* hexes = (uint8_t*)self.hexahedra;
//       i0 = hexes[ID * 8 + 0];
//       i1 = hexes[ID * 8 + 1];
//       i2 = hexes[ID * 8 + 2];
//       i3 = hexes[ID * 8 + 3];
//       i4 = hexes[ID * 8 + 4];
//       i5 = hexes[ID * 8 + 5];
//       i6 = hexes[ID * 8 + 6];
//       i7 = hexes[ID * 8 + 7];
//     } else if (self.bytesPerIndex == 2) {
//       uint16_t* hexes = (uint16_t*)self.hexahedra;
//       i0 = hexes[ID * 8 + 0];
//       i1 = hexes[ID * 8 + 1];
//       i2 = hexes[ID * 8 + 2];
//       i3 = hexes[ID * 8 + 3];
//       i4 = hexes[ID * 8 + 4];
//       i5 = hexes[ID * 8 + 5];
//       i6 = hexes[ID * 8 + 6];
//       i7 = hexes[ID * 8 + 7];
//     } else {
//       uint32_t* hexes = (uint32_t*)self.hexahedra;
//       i0 = hexes[ID * 8 + 0];
//       i1 = hexes[ID * 8 + 1];
//       i2 = hexes[ID * 8 + 2];
//       i3 = hexes[ID * 8 + 3];
//       i4 = hexes[ID * 8 + 4];
//       i5 = hexes[ID * 8 + 5];
//       i6 = hexes[ID * 8 + 6];
//       i7 = hexes[ID * 8 + 7];
//     }

//     vec3f P0 = self.vertices[i0];
//     vec3f P1 = self.vertices[i1];
//     vec3f P2 = self.vertices[i2];
//     vec3f P3 = self.vertices[i3];
//     vec3f P4 = self.vertices[i4];
//     vec3f P5 = self.vertices[i5];
//     vec3f P6 = self.vertices[i6];
//     vec3f P7 = self.vertices[i7];
//     primBounds = primBounds
//     .including(P0)
//     .including(P1)
//     .including(P2)
//     .including(P3)
//     .including(P4)
//     .including(P5)
//     .including(P6)
//     .including(P7);
//   }
// }

// ------------------------------------------------------------------
// intersection programs
// ------------------------------------------------------------------

OPTIX_INTERSECT_PROGRAM(VolumeIntersection)
()
{
  // auto &lp = optixLaunchParams;
  RayPayload &prd = owl::getPRD<RayPayload>();
  const auto &self = owl::getProgramData<MacrocellData>();
  const int primID = optixGetPrimitiveIndex();

  // // avoid intersecting the same brick twice
  // // if (primID == prd.prevNode) return;

  // if (prd.rgba.w > .99f)
  //   return;

  box4f bbox;
  bbox.extend(self.bboxes[primID]).extend(self.bboxes[primID + 1]);
  float3 lb = make_float3(bbox.lower.x, bbox.lower.y, bbox.lower.z);
  float3 rt = make_float3(bbox.upper.x, bbox.upper.y, bbox.upper.z);
  float3 origin = optixGetObjectRayOrigin();
  // note, this is _not_ normalized. Useful for computing world space tmin/mmax
  float3 direction = optixGetObjectRayDirection();
  vec3f dir = vec3f(direction.x, direction.y, direction.z);
  dir = normalize(dir);
  float3 dirfrac;

  if (prd.debug)
  {
    printf("bbox: min %f %f %f\n\t max %f %f %f\n",
           bbox.lower.x, bbox.lower.y, bbox.upper.z,
           bbox.upper.x, bbox.upper.y, bbox.lower.z);
  }

  // direction is unit direction vector of ray
  dirfrac.x = 1.0f / dir.x;
  dirfrac.y = 1.0f / dir.y;
  dirfrac.z = 1.0f / dir.z;

  // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
  float t0x = (lb.x - origin.x) * dirfrac.x;
  float t1x = (rt.x - origin.x) * dirfrac.x;
  float t0y = (lb.y - origin.y) * dirfrac.y;
  float t1y = (rt.y - origin.y) * dirfrac.y;
  float t0z = (lb.z - origin.z) * dirfrac.z;
  float t1z = (rt.z - origin.z) * dirfrac.z;

  float3 tmin = make_float3(min(t0x, t1x), min(t0y, t1y), min(t0z, t1z));
  float3 tmax = make_float3(max(t0x, t1x), max(t0y, t1y), max(t0z, t1z));

  float tNear = max(max(tmin.x, tmin.y), tmin.z);
  float tFar = min(min(tmax.x, tmax.y), tmax.z);

  if (prd.debug)
  {
    printf("ray: %f %f %f -> %f %f %f\n\t t0 %f %f %f, t1 %f %f %f\n\t tmin %f %f %f\n\t tmax %f %f %f\n\t tNear %f tFar %f\n",
           origin.x, origin.y, origin.z,
           direction.x, direction.y, direction.z,
           t0x, t0y, t0z, t1x, t1y, t1z,
           tmin.x, tmin.y, tmin.z,
           tmax.x, tmax.y, tmax.z,
           tNear, tFar);
  }

  // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
  if (tFar < 0)
  {
    // tFar = -tFar;
    return;
  }

  // if tmin > tmax, ray doesn't intersect AABB
  if (tNear > tFar)
  {
    float tmp = tNear;
    tNear = tFar;
    tFar = tmp;
    // return;
  }

  // clip hit to near position
  tNear = max(tNear, optixGetRayTmin());

  if (optixReportIntersection(tNear, /* hit kind */ 0))
  {
    prd.t0 = max(prd.t0, tNear);
    prd.t1 = min(prd.t1, tFar);
    // prd.rng.init(bbox.lower.w, bbox.upper.w);
    //  prd.rgba = make_float4(prd.rng(), prd.rng(), prd.rng(), 1.f);
    prd.dataValue = (bbox.lower.w + bbox.upper.w) * 0.5f;
  }
}

OPTIX_INTERSECT_PROGRAM(TetrahedraPointQuery)
()
{
  RayPayload &prd = owl::getPRD<RayPayload>();
  const auto &self = owl::getProgramData<UnstructuredElementData>();
  unsigned int primID = optixGetPrimitiveIndex(); //+ self.offset;
  float3 origin = optixGetObjectRayOrigin();

  // for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
  //   uint32_t ID = primID * ELEMENTS_PER_BOX + i;
  if (primID >= self.numTetrahedra)
    return;

  // printf("TetrahedraPointQuery: primID = %d\\n", primID);

  unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;

  vec3f P = {origin.x, origin.y, origin.z};

  // unsigned int i0, i1, i2, i3;
  uint32_t *tets = (uint32_t *)self.tetrahedra;
  uint64_t i0 = tets[ID * 4 + 0];
  uint64_t i1 = tets[ID * 4 + 1];
  uint64_t i2 = tets[ID * 4 + 2];
  uint64_t i3 = tets[ID * 4 + 3];

  vec3f P0 = self.vertices[i0];
  vec3f P1 = self.vertices[i1];
  vec3f P2 = self.vertices[i2];
  vec3f P3 = self.vertices[i3];

  float S0 = self.scalars[i0];
  float S1 = self.scalars[i1];
  float S2 = self.scalars[i2];
  float S3 = self.scalars[i3];

  prd.missed = false;              // for
  prd.dataValue = S0;              // testing
  optixReportIntersection(0.f, 0); // please
  return;                          // remove

  if (interpolateTetrahedra(P, P0, P1, P2, P3, S0, S1, S2, S3, prd.dataValue))
  {
    optixReportIntersection(0.f, 0);
    prd.missed = false;
    return;
  }
}

OPTIX_INTERSECT_PROGRAM(PyramidPointQuery)
()
{
  RayPayload &prd = owl::getPRD<RayPayload>();
  const auto &self = owl::getProgramData<UnstructuredElementData>();
  uint64_t primID = optixGetPrimitiveIndex() + self.offset;
  float3 origin = optixGetObjectRayOrigin();

  // float maxima = self.maxima[self.numTetrahedra + primID];
  // if (maxima <= 0.f) return;

  vec3f P = {origin.x, origin.y, origin.z};
  for (int i = 0; i < ELEMENTS_PER_BOX; ++i)
  {
    uint32_t ID = primID * ELEMENTS_PER_BOX + i;
    if (ID >= self.numPyramids)
      return;

    uint64_t i0, i1, i2, i3, i4;
    if (self.bytesPerIndex == 1)
    {
      uint8_t *pyrs = (uint8_t *)self.pyramids;
      i0 = pyrs[ID * 5 + 0];
      i1 = pyrs[ID * 5 + 1];
      i2 = pyrs[ID * 5 + 2];
      i3 = pyrs[ID * 5 + 3];
      i4 = pyrs[ID * 5 + 4];
    }
    else if (self.bytesPerIndex == 2)
    {
      uint16_t *pyrs = (uint16_t *)self.pyramids;
      i0 = pyrs[ID * 5 + 0];
      i1 = pyrs[ID * 5 + 1];
      i2 = pyrs[ID * 5 + 2];
      i3 = pyrs[ID * 5 + 3];
      i4 = pyrs[ID * 5 + 4];
    }
    else
    {
      uint32_t *pyrs = (uint32_t *)self.pyramids;
      i0 = pyrs[ID * 5 + 0];
      i1 = pyrs[ID * 5 + 1];
      i2 = pyrs[ID * 5 + 2];
      i3 = pyrs[ID * 5 + 3];
      i4 = pyrs[ID * 5 + 4];
    }

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];
    vec3f P4 = self.vertices[i4];

    float S0 = self.scalars[i0];
    float S1 = self.scalars[i1];
    float S2 = self.scalars[i2];
    float S3 = self.scalars[i3];
    float S4 = self.scalars[i4];

    if (interpolatePyramid(P, P0, P1, P2, P3, P4, S0, S1, S2, S3, S4, prd.dataValue))
    {
      optixReportIntersection(0.f, 0);
      return;
    }
  }
}

OPTIX_INTERSECT_PROGRAM(WedgePointQuery)
()
{
  RayPayload &prd = owl::getPRD<RayPayload>();
  const auto &self = owl::getProgramData<UnstructuredElementData>();
  uint64_t primID = optixGetPrimitiveIndex() + self.offset;
  float3 origin = optixGetObjectRayOrigin();

  // float maxima = self.maxima[self.numTetrahedra + self.numPyramids + primID];
  // if (maxima <= 0.f) return;

  vec3f P = {origin.x, origin.y, origin.z};

  // primID -= wedOffset;

  for (int i = 0; i < ELEMENTS_PER_BOX; ++i)
  {
    uint32_t ID = primID * ELEMENTS_PER_BOX + i;
    if (ID >= self.numWedges)
      return;

    uint64_t i0, i1, i2, i3, i4, i5;
    if (self.bytesPerIndex == 1)
    {
      uint8_t *wed = (uint8_t *)self.wedges;
      i0 = wed[ID * 6 + 0];
      i1 = wed[ID * 6 + 1];
      i2 = wed[ID * 6 + 2];
      i3 = wed[ID * 6 + 3];
      i4 = wed[ID * 6 + 4];
      i5 = wed[ID * 6 + 5];
    }
    else if (self.bytesPerIndex == 2)
    {
      uint16_t *wed = (uint16_t *)self.wedges;
      i0 = wed[ID * 6 + 0];
      i1 = wed[ID * 6 + 1];
      i2 = wed[ID * 6 + 2];
      i3 = wed[ID * 6 + 3];
      i4 = wed[ID * 6 + 4];
      i5 = wed[ID * 6 + 5];
    }
    else
    {
      uint32_t *wed = (uint32_t *)self.wedges;
      i0 = wed[ID * 6 + 0];
      i1 = wed[ID * 6 + 1];
      i2 = wed[ID * 6 + 2];
      i3 = wed[ID * 6 + 3];
      i4 = wed[ID * 6 + 4];
      i5 = wed[ID * 6 + 5];
    }

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];
    vec3f P4 = self.vertices[i4];
    vec3f P5 = self.vertices[i5];

    float S0 = self.scalars[i0];
    float S1 = self.scalars[i1];
    float S2 = self.scalars[i2];
    float S3 = self.scalars[i3];
    float S4 = self.scalars[i4];
    float S5 = self.scalars[i5];

    if (interpolateWedge(P, P0, P1, P2, P3, P4, P5, S0, S1, S2, S3, S4, S5, prd.dataValue))
    {
      optixReportIntersection(0.f, 0);
      return;
    }
  }
}

OPTIX_INTERSECT_PROGRAM(HexahedraPointQuery)
()
{
  RayPayload &prd = owl::getPRD<RayPayload>();
  const auto &self = owl::getProgramData<UnstructuredElementData>();
  uint64_t primID = optixGetPrimitiveIndex() + self.offset;
  float3 origin = optixGetObjectRayOrigin();

  // float maxima = self.maxima[self.numTetrahedra + self.numPyramids + self.numWedges + primID];
  // if (maxima <= 0.f) return;

  vec3f P = {origin.x, origin.y, origin.z};

  // primID -= hexOffset;
  for (int i = 0; i < ELEMENTS_PER_BOX; ++i)
  {
    uint32_t ID = primID * ELEMENTS_PER_BOX + i;
    if (ID >= self.numHexahedra)
      return;

    uint64_t i0, i1, i2, i3, i4, i5, i6, i7;
    if (self.bytesPerIndex == 1)
    {
      uint8_t *hexes = (uint8_t *)self.hexahedra;
      i0 = hexes[ID * 8 + 0];
      i1 = hexes[ID * 8 + 1];
      i2 = hexes[ID * 8 + 2];
      i3 = hexes[ID * 8 + 3];
      i4 = hexes[ID * 8 + 4];
      i5 = hexes[ID * 8 + 5];
      i6 = hexes[ID * 8 + 6];
      i7 = hexes[ID * 8 + 7];
    }
    else if (self.bytesPerIndex == 2)
    {
      uint16_t *hexes = (uint16_t *)self.hexahedra;
      i0 = hexes[ID * 8 + 0];
      i1 = hexes[ID * 8 + 1];
      i2 = hexes[ID * 8 + 2];
      i3 = hexes[ID * 8 + 3];
      i4 = hexes[ID * 8 + 4];
      i5 = hexes[ID * 8 + 5];
      i6 = hexes[ID * 8 + 6];
      i7 = hexes[ID * 8 + 7];
    }
    else
    {
      uint32_t *hexes = (uint32_t *)self.hexahedra;
      i0 = hexes[ID * 8 + 0];
      i1 = hexes[ID * 8 + 1];
      i2 = hexes[ID * 8 + 2];
      i3 = hexes[ID * 8 + 3];
      i4 = hexes[ID * 8 + 4];
      i5 = hexes[ID * 8 + 5];
      i6 = hexes[ID * 8 + 6];
      i7 = hexes[ID * 8 + 7];
    }

    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];
    vec3f P4 = self.vertices[i4];
    vec3f P5 = self.vertices[i5];
    vec3f P6 = self.vertices[i6];
    vec3f P7 = self.vertices[i7];

    float S0 = self.scalars[i0];
    float S1 = self.scalars[i1];
    float S2 = self.scalars[i2];
    float S3 = self.scalars[i3];
    float S4 = self.scalars[i4];
    float S5 = self.scalars[i5];
    float S6 = self.scalars[i6];
    float S7 = self.scalars[i7];

    if (interpolateHexahedra(P, P0, P1, P2, P3, P4, P5, P6, P7, S0, S1, S2, S3, S4, S5, S6, S7, prd.dataValue))
    {
      optixReportIntersection(0.f, 0);
      return;
    }
  }
}
