#include "DeviceCode.h"
#include <optix_device.h>

using namespace deltaVis;
using namespace owl;

extern "C" __constant__ LaunchParams optixLaunchParams;

#define DEBUG 1
// create a debug function macro that gets called only for center pixel
inline __device__ bool dbg()
{
  auto lp = optixLaunchParams;
#if DEBUG
  return false;
#else
  auto pixelID = vec2i(owl::getLaunchIndex()[0], owl::getLaunchIndex()[1]);
  return (lp.fbSize.x / 2 == pixelID.x) &&
         (lp.fbSize.y / 2 == pixelID.y);
#define ACTIVATE_CROSSHAIRS
#endif
}

__forceinline__ __device__ int optixGetPayload(int i)
{
  switch (i)
  {
  case 0:
    return optixGetPayload_0();
  case 1:
    return optixGetPayload_1();
  case 2:
    return optixGetPayload_2();
  case 3:
    return optixGetPayload_3();
  case 4:
    return optixGetPayload_4();
  case 5:
    return optixGetPayload_5();
  case 6:
    return optixGetPayload_6();
  case 7:
    return optixGetPayload_7();
  case 8:
    return optixGetPayload_8();
  case 9:
    return optixGetPayload_9();
  case 10:
    return optixGetPayload_10();
  case 11:
    return optixGetPayload_11();
  case 12:
    return optixGetPayload_12();
  case 13:
    return optixGetPayload_13();
  case 14:
    return optixGetPayload_14();
  case 15:
    return optixGetPayload_15();
  case 16:
    return optixGetPayload_16();
  case 17:
    return optixGetPayload_17();
  case 18:
    return optixGetPayload_18();
  case 19:
    return optixGetPayload_19();
  case 20:
    return optixGetPayload_20();
  case 21:
    return optixGetPayload_21();
  case 22:
    return optixGetPayload_22();
  case 23:
    return optixGetPayload_23();
  case 24:
    return optixGetPayload_24();
  case 25:
    return optixGetPayload_25();
  case 26:
    return optixGetPayload_26();
  case 27:
    return optixGetPayload_27();
  case 28:
    return optixGetPayload_28();
  case 29:
    return optixGetPayload_29();
  case 30:
    return optixGetPayload_30();
  case 31:
    return optixGetPayload_31();
  default:
    return 0;
  }
}

__forceinline__ __device__ void optixSetPayload(int i, int data)
{
  switch (i)
  {
  case 0:
    optixSetPayload_0(data);
    break;
  case 1:
    optixSetPayload_1(data);
    break;
  case 2:
    optixSetPayload_2(data);
    break;
  case 3:
    optixSetPayload_3(data);
    break;
  case 4:
    optixSetPayload_4(data);
    break;
  case 5:
    optixSetPayload_5(data);
    break;
  case 6:
    optixSetPayload_6(data);
    break;
  case 7:
    optixSetPayload_7(data);
    break;
  case 8:
    optixSetPayload_8(data);
    break;
  case 9:
    optixSetPayload_9(data);
    break;
  case 10:
    optixSetPayload_10(data);
    break;
  case 11:
    optixSetPayload_11(data);
    break;
  case 12:
    optixSetPayload_12(data);
    break;
  case 13:
    optixSetPayload_13(data);
    break;
  case 14:
    optixSetPayload_14(data);
    break;
  case 15:
    optixSetPayload_15(data);
    break;
  case 16:
    optixSetPayload_16(data);
    break;
  case 17:
    optixSetPayload_17(data);
    break;
  case 18:
    optixSetPayload_18(data);
    break;
  case 19:
    optixSetPayload_19(data);
    break;
  case 20:
    optixSetPayload_20(data);
    break;
  case 21:
    optixSetPayload_21(data);
    break;
  case 22:
    optixSetPayload_22(data);
    break;
  case 23:
    optixSetPayload_23(data);
    break;
  case 24:
    optixSetPayload_24(data);
    break;
  case 25:
    optixSetPayload_25(data);
    break;
  case 26:
    optixSetPayload_26(data);
    break;
  case 27:
    optixSetPayload_27(data);
    break;
  case 28:
    optixSetPayload_28(data);
    break;
  case 29:
    optixSetPayload_29(data);
    break;
  case 30:
    optixSetPayload_30(data);
    break;
  case 31:
    optixSetPayload_31(data);
    break;
  default:
    break;
  }
}

inline __device__ float4 missColor()
{
  return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}

// inline __device__ vec4f missColor()
// {
//   return vec4f(0.0f, 0.0f, 0.01f, 1.0f);
// }

//--------------------Math functions---------------------------
inline __device__ float4 make_float4(const vec4f &v)
{
  return make_float4(v.x, v.y, v.z, v.w);
}

inline __device__ float3 make_float3(const vec3f &v)
{
  return make_float3(v.x, v.y, v.z);
}

inline __device__ float2 make_float2(const vec2f &v)
{
  return make_float2(v.x, v.y);
}

inline __device__ vec4f make_vec4f(const float4 &v)
{
  return vec4f(v.x, v.y, v.z, v.w);
}

inline __device__ vec3f make_vec3f(const float3 &v)
{
  return vec3f(v.x, v.y, v.z);
}

inline __device__ vec2f make_vec2f(const float2 &v)
{
  return vec2f(v.x, v.y);
}

inline __device__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(const float3 &a, const float3 &b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator*(const float3 &a, const float3 &b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ float3 operator*(const float3 &a, const float &b)
{
  return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float3 operator*(const float &a, const float3 &b)
{
  return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __device__ float3 operator/(const float3 &a, const float3 &b)
{
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __device__ float3 operator/(const float3 &a, const float &b)
{
  return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __device__ float3 operator/(const float &a, const float3 &b)
{
  return make_float3(a / b.x, a / b.y, a / b.z);
}

inline __device__ float3 operator-(const float3 &a)
{
  return make_float3(-a.x, -a.y, -a.z);
}
//-------------------------------------------------------------

inline __device__
    float4
    transferFunction(float f)
{
  auto &lp = optixLaunchParams;
  if (f < lp.transferFunction.volumeDomain.x ||
      f > lp.transferFunction.volumeDomain.y)
  {
    return make_float4(1.f, 0.f, 1.f, 0.0f);
  }
  float remapped = (f - lp.transferFunction.volumeDomain.x) /
                   (lp.transferFunction.volumeDomain.y - lp.transferFunction.volumeDomain.x);

  float4 xf = tex2D<float4>(lp.transferFunction.xf, remapped, 0.5f);
  xf.w *= lp.transferFunction.opacityScale;

  return xf;
}

inline __device__ vec3f over(vec3f Cin, vec3f Cx, float Ain, float Ax)
{
  return Cin + Cx * Ax * (1.f - Ain);
}

inline __device__ float over(const float Ain, const float Ax)
{
  return Ain + (1.f - Ain) * Ax;
}

inline __device__ vec4f over(const vec4f &in, const vec4f &x)
{
  auto c = over(vec3f(in), vec3f(x), in.w, x.w);
  auto a = over(in.w, x.w);
  return vec4f(c, a);
}

inline __device__ void generateRay(const vec2f screen, owl::Ray &ray)
{
  auto &lp = optixLaunchParams;
  ray.origin = lp.camera.origin;
  vec3f direction = lp.camera.lower_left_corner +
                    screen.u * lp.camera.horizontal +
                    screen.v * lp.camera.vertical;
  // direction = normalize(direction);
  if (fabs(direction.x) < 1e-5f)
    direction.x = 1e-5f;
  if (fabs(direction.y) < 1e-5f)
    direction.y = 1e-5f;
  if (fabs(direction.z) < 1e-5f)
    direction.z = 1e-5f;
  ray.direction = normalize(direction - ray.origin);
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)
()
{
  auto &lp = optixLaunchParams;
  const vec2i pixelID = owl::getLaunchIndex();
  // if (pixelID == owl::vec2i(0)) {
  //   printf("%sHello OptiX From your First RayGen Program%s\n",
  //          OWL_TERMINAL_CYAN,
  //          OWL_TERMINAL_DEFAULT);
  // }
  int seed = owl::getLaunchDims().x * owl::getLaunchDims().y * lp.frameID;
  owl::common::LCG<4> random(threadIdx.x + seed, threadIdx.y + seed);
  const vec2f screen = (vec2f(pixelID) + vec2f(.5f)) / vec2f(lp.fbSize);
  owl::Ray ray;
  generateRay(screen, ray);
  // ray.origin = ray.origin + random() * lp.camera.horizontal * 0.0005f +
  //              random() * lp.camera.vertical * 0.0005f;

  RayPayload prd;
  //prd.missed = true;
  // prd.rgba = vec4f(0, 0, 0, 0);
  // prd.dataValue = 0;
  prd.debug = dbg();
  // prd.t0 = 0.f;
  // prd.t1 = 1e20f;
  // prd.tHit = 1e20f;
  // owl::traceRay(/*accel to trace against*/ lp.volume.macrocellTLAS,
  //               /*the ray to trace*/ ray,
  //               /*prd*/ prd);

  prd.t0 = 0.f;
  prd.t1 = 1e20f;
  prd.tHit = 1e20f;
  prd.rgba = vec4f(0.f);
  prd.shadowRay = false;
  //prd.random = random;

  unsigned int p0 = 0;
  unsigned int p1 = 0;
  owl::packPointer(&prd, p0, p1);

  // Volume Integration Ray
  optixTrace(lp.volume.macrocellTLAS,
             (const float3 &)ray.origin,
             (const float3 &)ray.direction,
             0.f, 1e20f, 0.f,
             ray.visibilityMask, //(OptixVisibilityMask)-1,
             /*rayFlags     */ OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             /*SBToffset    */ 0 /* ray type */,
             /*SBTstride    */ 2 /* num ray types */,
             /*missSBTIndex */ 0,
             p0,
             p1);
  vec3f albedo = vec3f(prd.rgba);
  float albedo_alpha = prd.rgba.w;

  if (lp.shadows)
  {
    owl::Ray shadowRay;
    shadowRay.origin = ray.origin + ray.direction * prd.tHit;
    shadowRay.direction = -lp.lightDir;
    RayPayload shadowPrd;
    shadowPrd.missed = true;
    shadowPrd.rgba = vec4f(0, 0, 0, 0);
    shadowPrd.dataValue = 0;
    shadowPrd.debug = dbg();
    shadowPrd.t0 = 0.0f;
    shadowPrd.t1 = 1e20f;
    shadowPrd.tHit = 1e20f;
    owl::traceRay(/*accel to trace against*/ lp.volume.macrocellTLAS,
                  /*the ray to trace*/ shadowRay,
                  /*prd*/ shadowPrd);
    if (shadowPrd.missed)
    {
      shadowPrd.rgba.w = 0.0f;
    }
    vec3f shadow = vec3f((1.f - 0.2f) * (1.f - shadowPrd.rgba.w) + 0.2f);
    float light_intensity = 2.0f;
    albedo = albedo * shadow * light_intensity;
  }

  // map prd.dataValue to color
  // float4 tfColor = transferFunction(prd.dataValue);

  vec4f color = vec4f(albedo, albedo_alpha); // prd.missed ? prd.rgba : vec4f(tfColor.x, tfColor.y, tfColor.z, tfColor.w);

  // vec3f color = vec3f(prd.rgba.x, prd.rgba.y, prd.rgba.z);
  //color = over(color, vec4f(owl::getProgramData<MissProgData>().color1, 1.0f));
  const int fbOfs = pixelID.x + lp.fbSize.x * pixelID.y;
  // lp.fbPtr[fbOfs] = owl::make_rgba(color);
  vec4f oldColor = lp.accumBuffer[fbOfs];
  vec4f newColor = (vec4f(color) + float(lp.accumID) * oldColor) / float(lp.accumID + 1);
  lp.fbPtr[fbOfs] = make_rgba(vec4f(newColor));
  lp.accumBuffer[fbOfs] = vec4f(newColor);

#ifdef ACTIVATE_CROSSHAIRS
  if (pixelID.x == lp.fbSize.x / 2 || pixelID.y == lp.fbSize.y / 2 ||
      pixelID.x == lp.fbSize.x / 2 + 1 || pixelID.y == lp.fbSize.y / 2 + 1 ||
      pixelID.x == lp.fbSize.x / 2 - 1 || pixelID.y == lp.fbSize.y / 2 - 1)
    lp.fbPtr[fbOfs] = owl::make_rgba(1.f-color * 0.33f);
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
  const MacrocellData &self = owl::getProgramData<MacrocellData>();
  RayPayload &prd = owl::getPRD<RayPayload>();
  auto &lp = optixLaunchParams;
  vec3f origin = vec3f(optixGetWorldRayOrigin());
  vec3f direction = vec3f(optixGetWorldRayDirection());
  prd.missed = true;
  prd.rgba = missColor();

  // auto sampler = Sampler(prd.debug);
  owl::Ray ray;
  ray.origin = origin;
  ray.direction = {1, 1, 1};
  prd.dataValue = NAN;

  float majorantExtinction = transferFunction(self.bboxes[1].w).w;
  // normalize the majorant
  // majorantExtinction = (majorantExtinction - lp.transferFunction.volumeDomain.x) /
  //                      (lp.transferFunction.volumeDomain.y - lp.transferFunction.volumeDomain.x);

  // majorantExtinction = 1.4f;
  if (majorantExtinction == 0.f)
    return;

  float unit = max(lp.volume.dt, 0.0001f);
  float t = prd.t0;
  for (int i = 0; i < 100000; ++i)
  {
    // Sample a distance
    t = t - (log(1.0f - prd.rng()) / majorantExtinction) * unit;

    // A boundary has been hit
    if (t >= prd.t1)
      break;

    // Update current position
    vec3f x = origin + t * direction;

    //-----Sample heterogeneous media-----
    owl::Ray ray;
    ray.origin = x;
    ray.direction = {1, 1, 1};
    prd.dataValue = NAN;

    owl::traceRay(lp.volume.elementTLAS, ray, prd,
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);

    //------------------------------------
    float4 xf ={0,0,0,0}; //missColor();
    if (!isnan(prd.dataValue))
    {
      xf = transferFunction(prd.dataValue);
      prd.dataValue = prd.dataValue;
    }

    if (prd.debug)
      printf("t0 %f t %f t1 %f, majorantExtinction: %f xf.w %f\n",
             prd.t0, t, prd.t1, majorantExtinction, xf.w);

    // Check if an emission occurred
    if (prd.rng() < xf.w / (majorantExtinction))
    {
      prd.tHit = min(prd.tHit, t);
      prd.rgba = vec4f(vec3f(xf), 1.f);
      prd.missed = false;
      break;
    }
  }
  if (prd.debug)
    printf("rgba: %f %f %f %f, tHit: %f dataValue: %f\n",
           prd.rgba.x, prd.rgba.y, prd.rgba.z, prd.rgba.w, prd.tHit, prd.dataValue);
}

OPTIX_CLOSEST_HIT_PROGRAM(AdaptiveDeltaTracking)
()
{
  const MacrocellData &self = owl::getProgramData<MacrocellData>();
  RayPayload &prd = owl::getPRD<RayPayload>();
  auto &lp = optixLaunchParams;
  int numAdaptiveRays = 32; // lp.volume.numAdaptiveSamplingRays;
  //printf("called adaptive delta tracking\n");
  for (int asi = 0; asi < numAdaptiveRays; ++asi)
  {
    float t00 = (prd.t1 - prd.t0) * (float(asi + 0.f) / float(numAdaptiveRays)) + prd.t0;
    float t11 = (prd.t1 - prd.t0) * (float(asi + 1.f) / float(numAdaptiveRays)) + prd.t0;

    // Coarse Adaptive Sampling Ray
    unsigned int r1[NUM_BINS] = {0};
    optixTrace(lp.volume.macrocellTLAS,
               optixGetWorldRayOrigin(),
               optixGetWorldRayDirection(),
               t00, t11, 0.f,
               (OptixVisibilityMask)-1,
               /*rayFlags     */ OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               /*SBToffset    */ 1 /* ray type */,
               /*SBTstride    */ 2 /* num ray types */,
               /*missSBTIndex */ 1,
#if NUM_BINS == 8
               r1[0], r1[1], r1[2], r1[3], r1[4], r1[5], r1[6], r1[7]
#elif NUM_BINS == 16
               r1[0], r1[1], r1[2], r1[3], r1[4], r1[5], r1[6], r1[7], r1[8], r1[9], r1[10], r1[11], r1[12], r1[13], r1[14], r1[15]
#elif NUM_BINS == 32
               r1[0], r1[1], r1[2], r1[3], r1[4], r1[5], r1[6], r1[7], r1[8], r1[9], r1[10], r1[11], r1[12], r1[13], r1[14], r1[15], r1[16], r1[17], r1[18], r1[19], r1[20], r1[21], r1[22], r1[23], r1[24], r1[25], r1[26], r1[27], r1[28], r1[29], r1[30], r1[31]
#endif
    );

    // Move ray to the volume boundary
    vec3f origin = vec3f(optixGetWorldRayOrigin());
    vec3f direction = vec3f(optixGetWorldRayDirection());

    vec3f color = vec3f(0.f);
    int event;

    float4 xf = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int i = 0; i < NUM_BINS; ++i)
    {
      // if bin is empty, skip it.
      float majorantExtinction = __int_as_float(r1[i]);
      if (majorantExtinction == 0.f)
        continue;

      float t0 = t00 + (float(i + 0) / float(NUM_BINS)) * (t11 - t00);
      float t1 = t00 + (float(i + 1) / float(NUM_BINS)) * (t11 - t00);

      // Sample free-flight distance
      float t = t0;
      float unit = 0.1f;
      while (true)
      {
        t = t - (log(1.0f - prd.rng()) / majorantExtinction) * unit;

        // A boundary has been hit
        if (t >= t1)
        {
          event = 0;
          break;
        }

        // Update current position
        vec3f x = origin + t * direction;

        float eventRand = prd.rng();

        //-----Sample heterogeneous media-----
        owl::Ray ray;
        ray.origin = x;
        ray.direction = {1, 1, 1};
        RayPayload samplerPrd;
        samplerPrd.dataValue = 0.f;

        owl::traceRay(lp.volume.elementTLAS, ray, samplerPrd,
                      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT);

        // prd.samples++;
        //------------------------------------
        if (samplerPrd.dataValue == 0.0f) //(dataValue == sampler.background())
        {
          xf = make_float4(1.f, 0.f, 1.f, 0.f);
        }
        else
        {
          xf = make_float4(transferFunction(samplerPrd.dataValue));
        }

        float absorption = xf.w;
        float scattering = 0.f;
        float extinction = absorption + scattering;
        float nullCollision = majorantExtinction - extinction;

        float pa = absorption / (extinction + abs(nullCollision));
        float ps = scattering / (extinction + abs(nullCollision));
        float pn = abs(nullCollision) / (extinction + abs(nullCollision));

        // An absorption / emission collision occurred
        if (eventRand < pa)
        {
          prd.tHit = min(prd.tHit, t);
          event = 1;
          break;
        }

        // A scattering collision occurred
        else if (eventRand < (ps + pa))
        {
          event = 2;
          break;
        }

        // A null collision occurred
        else
        {
          event = 3;
        }
      }

      // we hit the boundary, or last collision was a null collision
      if (event == 0)
      {
        // prd.rgba = vec4f(0.f, 0.f, 0.f, 0.f);
        continue;
      }
      // emission / absorption
      else if (event == 1)
      {
        auto c = vec3f(xf);
        prd.rgba = vec4f(c.x, c.y, c.z, 1.f);
        break;
      }
      // scattering
      else if (event == 2 || event == 42)
      {
        prd.rgba = vec4f(0.f, 1.f, 1.f, 1.f); // shouldn't happen with the current configuration
        break;
      }

      // if we hit, we're done, so stop adaptive sampling
      if (event != 0)
      {
        prd.missed = false;
        break;
      }
    }

    // if we hit, we're done, so stop adaptive sampling
    if (event != 0)
    {
      prd.missed = false;
      break;
    }
  }
}

OPTIX_MISS_PROGRAM(miss)
()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();

  RayPayload &prd = owl::getPRD<RayPayload>();
  // int pattern = (pixelID.x / 18) ^ (pixelID.y / 18);
  prd.rgba = missColor(); // vec4f(self.color1, 1); //(pattern & 1) ? vec4f(self.color1, 1) : vec4f(self.color0, 1);
  prd.missed = true;
}

// ------------------------------------------------------------------
// Bounds programs for volume elements
// ------------------------------------------------------------------

OPTIX_BOUNDS_PROGRAM(MacrocellBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID,
    const int key)
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
                                            self.bboxes[(primID * 2 + 0)].z));
    primBounds = primBounds.including(vec3f(self.bboxes[(primID * 2 + 1)].x,
                                            self.bboxes[(primID * 2 + 1)].y,
                                            self.bboxes[(primID * 2 + 1)].z));
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
    const int primID,
    const int key)
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
}

OPTIX_BOUNDS_PROGRAM(PyramidBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID,
    const int key)

{
  const UnstructuredElementData &self = *(const UnstructuredElementData *)geomData;
  primBounds = box3f();
  unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;
  if (ID >= self.numPyramids)
    return;

  unsigned int *pyrs = (unsigned int *)self.pyramids;
  uint64_t i0 = pyrs[ID * 5 + 0];
  uint64_t i1 = pyrs[ID * 5 + 1];
  uint64_t i2 = pyrs[ID * 5 + 2];
  uint64_t i3 = pyrs[ID * 5 + 3];
  uint64_t i4 = pyrs[ID * 5 + 4];

  vec3f P0 = self.vertices[i0];
  vec3f P1 = self.vertices[i1];
  vec3f P2 = self.vertices[i2];
  vec3f P3 = self.vertices[i3];
  vec3f P4 = self.vertices[i4];

  primBounds = primBounds.including(P0)
                   .including(P1)
                   .including(P2)
                   .including(P3)
                   .including(P4);
}

OPTIX_BOUNDS_PROGRAM(WedgeBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID,
    const int key)
{
  const UnstructuredElementData &self = *(const UnstructuredElementData *)geomData;
  primBounds = box3f();
  unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;
  if (ID >= self.numWedges)
    return;

  unsigned int *weds = (unsigned int *)self.wedges;
  uint64_t i0 = weds[ID * 6 + 0];
  uint64_t i1 = weds[ID * 6 + 1];
  uint64_t i2 = weds[ID * 6 + 2];
  uint64_t i3 = weds[ID * 6 + 3];
  uint64_t i4 = weds[ID * 6 + 4];
  uint64_t i5 = weds[ID * 6 + 5];

  vec3f P0 = self.vertices[i0];
  vec3f P1 = self.vertices[i1];
  vec3f P2 = self.vertices[i2];
  vec3f P3 = self.vertices[i3];
  vec3f P4 = self.vertices[i4];
  vec3f P5 = self.vertices[i5];

  primBounds = primBounds.including(P0)
                   .including(P1)
                   .including(P2)
                   .including(P3)
                   .including(P4)
                   .including(P5);
}

OPTIX_BOUNDS_PROGRAM(HexahedraBounds)
(
    const void *geomData,
    owl::common::box3f &primBounds,
    const int primID,
    const int key)
{
  const UnstructuredElementData &self = *(const UnstructuredElementData *)geomData;
  primBounds = box3f();
  unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;
  if (ID >= self.numHexahedra)
    return;

  unsigned int *hexes = (unsigned int *)self.hexahedra;
  uint64_t i0 = hexes[ID * 8 + 0];
  uint64_t i1 = hexes[ID * 8 + 1];
  uint64_t i2 = hexes[ID * 8 + 2];
  uint64_t i3 = hexes[ID * 8 + 3];
  uint64_t i4 = hexes[ID * 8 + 4];
  uint64_t i5 = hexes[ID * 8 + 5];
  uint64_t i6 = hexes[ID * 8 + 6];
  uint64_t i7 = hexes[ID * 8 + 7];

  vec3f P0 = self.vertices[i0];
  vec3f P1 = self.vertices[i1];
  vec3f P2 = self.vertices[i2];
  vec3f P3 = self.vertices[i3];
  vec3f P4 = self.vertices[i4];
  vec3f P5 = self.vertices[i5];
  vec3f P6 = self.vertices[i6];
  vec3f P7 = self.vertices[i7];
  primBounds.extend(P0)
            .extend(P1)
            .extend(P2)
            .extend(P3)
            .extend(P4)
            .extend(P5)
            .extend(P6)
            .extend(P7);
  //primBounds.extend(P7); // wtf??!
}

// ------------------------------------------------------------------
// intersection programs
// ------------------------------------------------------------------

OPTIX_INTERSECT_PROGRAM(MacrocellIntersection)
()
{
  //printf("called macrocell intersection program\n");
  const auto &self = owl::getProgramData<MacrocellData>();
  const int primID = optixGetPrimitiveIndex();
  float3 origin = optixGetObjectRayOrigin();

  // Get bbox max
  float maxima = transferFunction(self.bboxes[primID*2 + 1].w).w;

  /* Skip "splatting" empty bounds */
  if (maxima <= 0.f)
    return;

  // note, this is _not_ normalized. Useful for computing world space tmin/mmax
  float3 direction = optixGetObjectRayDirection();

  // typical ray AABB intersection test
  float3 dirfrac;

  // direction is unit direction vector of ray
  dirfrac.x = 1.0f / direction.x;
  dirfrac.y = 1.0f / direction.y;
  dirfrac.z = 1.0f / direction.z;

  // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
  // origin is origin of ray
  float4 rt = self.bboxes[primID * 2 + 1];
  float t2 = (rt.x - origin.x) * dirfrac.x;
  float t4 = (rt.y - origin.y) * dirfrac.y;
  float t6 = (rt.z - origin.z) * dirfrac.z;

  float4 lb = self.bboxes[primID * 2 + 0];
  float t1 = (lb.x - origin.x) * dirfrac.x;
  float t3 = (lb.y - origin.y) * dirfrac.y;
  float t5 = (lb.z - origin.z) * dirfrac.z;

  float thit0 = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
  float thit1 = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

  // clip hit to near position
  thit0 = max(thit0, optixGetRayTmin());
  thit1 = min(thit1, optixGetRayTmax());
  // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
  if (thit1 < 0)
  {
    return;
  }

  // if tmin > tmax, ray doesn't intersect AABB
  if (thit0 >= thit1)
  {
    return;
  }

  /* Map the hit min and max tmax to the occupancy bins */
  float fminhitaddr = (thit0 - optixGetRayTmin()) / (optixGetRayTmax() - optixGetRayTmin());
  float fmaxhitaddr = (thit1 - optixGetRayTmin()) / (optixGetRayTmax() - optixGetRayTmin());
  int minhitaddr = min(int(floor(fminhitaddr * (NUM_BINS))), NUM_BINS - 1);
  int maxhitaddr = min(int(ceil(fmaxhitaddr * (NUM_BINS))), NUM_BINS - 1);

  /* Splat that max value onto the bins */
  for (int i = minhitaddr; i <= maxhitaddr; ++i)
  {
    optixSetPayload(i, __float_as_int(max(__int_as_float(optixGetPayload(i)), maxima)));
  }
}

OPTIX_INTERSECT_PROGRAM(VolumeIntersection)
()
{
  auto &lp = optixLaunchParams;
  RayPayload &prd = owl::getPRD<RayPayload>();
  const auto &self = owl::getProgramData<MacrocellData>();
  const int primID = optixGetPrimitiveIndex();

  // // avoid intersecting the same brick twice
  // // if (primID == prd.prevNode) return;

  if (prd.rgba.w > .99f)
    return;

  box4f bbox;
  // bbox.lower = make_vec4f(lp.volume.globalBoundsLo);
  // bbox.upper = make_vec4f(lp.volume.globalBoundsHi);
  bbox.extend(self.bboxes[2 * primID]).extend(self.bboxes[2 * primID + 1]);
  float3 lb = make_float3(bbox.lower.x, bbox.lower.y, bbox.lower.z);
  float3 rt = make_float3(bbox.upper.x, bbox.upper.y, bbox.upper.z);
  float3 origin = optixGetObjectRayOrigin();
  // note, this is _not_ normalized. Useful for computing world space tmin/mmax
  float3 direction = optixGetObjectRayDirection();
  vec3f dir = vec3f(direction.x, direction.y, direction.z);
  dir = normalize(dir);
  float3 dirfrac;

  // if (prd.debug)
  // {
  //   printf("bbox: min %f %f %f\n\t max %f %f %f\n",
  //          bbox.lower.x, bbox.lower.y, bbox.upper.z,
  //          bbox.upper.x, bbox.upper.y, bbox.lower.z);
  // }

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
    return;
  }

  // if tmin > tmax, ray doesn't intersect AABB
  if (tNear > tFar)
  {
    return;
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

  // prd.missed = false;              // for
  // prd.dataValue = S0;              // testing
  // optixReportIntersection(0.f, 0); // please
  // return;                          // remove

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
  unsigned int primID = optixGetPrimitiveIndex(); //+ self.offset;
  float3 origin = optixGetObjectRayOrigin();

  // for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
  //   uint32_t ID = primID * ELEMENTS_PER_BOX + i;
  if (primID >= self.numPyramids)
    return;

  // printf("TetrahedraPointQuery: primID = %d\\n", primID);

  unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;

  vec3f P = {origin.x, origin.y, origin.z};

  // unsigned int i0, i1, i2, i3;
  uint32_t *pyrs = (uint32_t *)self.pyramids;
  uint64_t i0 = pyrs[ID * 5 + 0];
  uint64_t i1 = pyrs[ID * 5 + 1];
  uint64_t i2 = pyrs[ID * 5 + 2];
  uint64_t i3 = pyrs[ID * 5 + 3];
  uint64_t i4 = pyrs[ID * 5 + 4];

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

  // prd.missed = false;              // for
  // prd.dataValue = S0;              // testing
  // optixReportIntersection(0.f, 0); // please
  // return;                          // remove

  if (interpolatePyramid(P, P0, P1, P2, P3, P4, S0, S1, S2, S3, S4, prd.dataValue))
  {
    optixReportIntersection(0.f, 0);
    prd.missed = false;
    return;
  }
}

OPTIX_INTERSECT_PROGRAM(WedgePointQuery)
()
{
  RayPayload &prd = owl::getPRD<RayPayload>();
  const auto &self = owl::getProgramData<UnstructuredElementData>();
  unsigned int primID = optixGetPrimitiveIndex(); //+ self.offset;
  float3 origin = optixGetObjectRayOrigin();

  // for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
  //   uint32_t ID = primID * ELEMENTS_PER_BOX + i;
  if (primID >= self.numWedges)
    return;

  // printf("TetrahedraPointQuery: primID = %d\\n", primID);

  unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;

  vec3f P = {origin.x, origin.y, origin.z};

  // unsigned int i0, i1, i2, i3;
  uint32_t *weds = (uint32_t *)self.wedges;
  uint64_t i0 = weds[ID * 6 + 0];
  uint64_t i1 = weds[ID * 6 + 1];
  uint64_t i2 = weds[ID * 6 + 2];
  uint64_t i3 = weds[ID * 6 + 3];
  uint64_t i4 = weds[ID * 6 + 4];
  uint64_t i5 = weds[ID * 6 + 5];

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

  // prd.missed = false;              // for
  // prd.dataValue = S0;              // testing
  // optixReportIntersection(0.f, 0); // please
  // return;                          // remove

  if (interpolateWedge(P, P0, P1, P2, P3, P4, P5, S0, S1, S2, S3, S4, S5, prd.dataValue))
  {
    optixReportIntersection(0.f, 0);
    prd.missed = false;
    return;
  }
}

OPTIX_INTERSECT_PROGRAM(HexahedraPointQuery)
()
{
  RayPayload &prd = owl::getPRD<RayPayload>();
  const auto &self = owl::getProgramData<UnstructuredElementData>();
  unsigned int primID = optixGetPrimitiveIndex(); //+ self.offset;
  float3 origin = optixGetObjectRayOrigin();

  // for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
  //   uint32_t ID = primID * ELEMENTS_PER_BOX + i;
  if (primID >= self.numHexahedra)
    return;

  // printf("TetrahedraPointQuery: primID = %d\\n", primID);

  unsigned int ID = (uint32_t(primID)) /* ELEMENTS_PER_BOX*/;

  vec3f P = {origin.x, origin.y, origin.z};

  // unsigned int i0, i1, i2, i3;
  uint32_t *hexes = (uint32_t *)self.hexahedra;
  uint64_t i0 = hexes[ID * 8 + 0];
  uint64_t i1 = hexes[ID * 8 + 1];
  uint64_t i2 = hexes[ID * 8 + 2];
  uint64_t i3 = hexes[ID * 8 + 3];
  uint64_t i4 = hexes[ID * 8 + 4];
  uint64_t i5 = hexes[ID * 8 + 5];
  uint64_t i6 = hexes[ID * 8 + 6];
  uint64_t i7 = hexes[ID * 8 + 7];

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

  // prd.missed = false;              // for
  // prd.dataValue = S0;              // testing
  // optixReportIntersection(0.f, 0); // please
  // return;                          // remove

  if (interpolateHexahedra(P, P0, P1, P2, P3, P4, P5, P6, P7,
                           S0, S1, S2, S3, S4, S5, S6, S7, prd.dataValue))
  {
    optixReportIntersection(0.f, 0);
    prd.missed = false;
    return;
  }
}
