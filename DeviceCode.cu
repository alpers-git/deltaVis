#include "DeviceCode.h"
#include <optix_device.h>

using namespace deltaVis;
using namespace owl;

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  // if (pixelID == owl::vec2i(0)) {
  //   printf("%sHello OptiX From your First RayGen Program%s\n",
  //          OWL_TERMINAL_CYAN,
  //          OWL_TERMINAL_DEFAULT);
  // }

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  ray.origin    
    = self.camera.origin;
  const vec3f direction
      = self.camera.lower_left_corner
      + screen.u * self.camera.horizontal
      + screen.v * self.camera.vertical
      - self.camera.origin;
  ray.direction = normalize(direction);

  // vec3f color;
  // owl::traceRay(/*accel to trace against*/self.world,
  //               /*the ray to trace*/ray,
  //               /*prd*/color);
  RayPayload prd;
  float count = 0;
  while(true)
  {
    owl::traceRay(/*accel to trace against*/self.volume.elementTLAS,
                  /*the ray to trace*/ray,
                  /*prd*/prd);
    if(!prd.missed || count > 10.f)
      break;
    ray.origin += ray.direction * 0.1f;
    count += 1.f;
  }
  //map prd.dataValue to color
  vec3f color = prd.missed ? vec3f(prd.rgba.x, prd.rgba.y, prd.rgba.z) :
       vec3f(count/10.f, count/10.f, count/10.f);

  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleClosestHit)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TriangleData &self = owl::getProgramData<TriangleData>();
  
  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  const vec3i index  = self.indices[primID];
  const vec3f &A     = self.vertices[index.x];
  const vec3f &B     = self.vertices[index.y];
  const vec3f &C     = self.vertices[index.z];
  const vec3f Ng     = normalize(cross(B-A,C-A));

  const vec3f rayDir = optixGetWorldRayDirection();
  prd = (.2f + .8f*fabs(dot(rayDir,Ng)))*self.color;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  RayPayload &prd = owl::getPRD<RayPayload>();
  int pattern = (pixelID.x / 18) ^ (pixelID.y/18);
  prd.rgba = (pattern&1) ? vec4f(self.color1,1) : vec4f(self.color0,1);
  prd.missed = true;
}

// ------------------------------------------------------------------
// Bounds programs for volume elements
// ------------------------------------------------------------------
OPTIX_BOUNDS_PROGRAM(TetrahedraBounds)(
    const void  *geomData,
    owl::common::box3f &primBounds,
    const int    primID)
  {
    const UnstructuredElementData &self = *(const UnstructuredElementData*)geomData;
    primBounds = box3f();
    unsigned int ID = (uint32_t(primID) /*+ self.offset*/) /* ELEMENTS_PER_BOX*/;
    if (ID >= self.numTetrahedra) return;

    unsigned int* tets = (unsigned int*)self.tetrahedra;
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
    //printf("IDs: %d %d %d %d\n", i0, i1, i2, i3);
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

  OPTIX_INTERSECT_PROGRAM(TetrahedraPointQuery)()
  {
    RayPayload &prd = owl::getPRD<RayPayload>();
    const auto &self = owl::getProgramData<UnstructuredElementData>();
    uint64_t primID = optixGetPrimitiveIndex(); //+ self.offset; 
    float3 origin = optixGetObjectRayOrigin();

    // for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
    //   uint32_t ID = primID * ELEMENTS_PER_BOX + i;
    if (primID >= self.numTetrahedra) return;

    prd.missed = false;
    optixReportIntersection(0.f, 0);
    return;

    //printf("TetrahedraPointQuery: primID = %d\\n", primID);
  
    vec3f P = {origin.x, origin.y, origin.z};

    unsigned int i0, i1, i2, i3;
    unsigned int* tets = (unsigned int*)self.tetrahedra;
    i0 = tets[primID * 4 + 0];
    i1 = tets[primID * 4 + 1];
    i2 = tets[primID * 4 + 2];
    i3 = tets[primID * 4 + 3];
    
    vec3f P0 = self.vertices[i0];
    vec3f P1 = self.vertices[i1];
    vec3f P2 = self.vertices[i2];
    vec3f P3 = self.vertices[i3];

    float S0 = self.scalars[i0];
    float S1 = self.scalars[i1];
    float S2 = self.scalars[i2];
    float S3 = self.scalars[i3];

    if (interpolateTetrahedra(P,P0,P1,P2,P3,S0,S1,S2,S3,prd.dataValue)) {
      optixReportIntersection(0.f, 0);
      prd.missed = false;
      // printf("hit tetrahedron %d with value %f at (%f,%f,%f) with origin (%f,%f,%f)\n", 
      //     primID, prd.dataValue, P.x, P.y, P.z, origin.x, origin.y, origin.z);
      return;
    }
  }

  OPTIX_INTERSECT_PROGRAM(PyramidPointQuery)()
  {
    RayPayload &prd = owl::getPRD<RayPayload>();
    const auto &self = owl::getProgramData<UnstructuredElementData>();
    uint64_t primID = optixGetPrimitiveIndex() + self.offset; 
    float3 origin = optixGetObjectRayOrigin();

    // float maxima = self.maxima[self.numTetrahedra + primID];
    // if (maxima <= 0.f) return;
   
    vec3f P = {origin.x, origin.y, origin.z};
    for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
      uint32_t ID = primID * ELEMENTS_PER_BOX + i;
      if (ID >= self.numPyramids) return;

      uint64_t i0, i1, i2, i3, i4;
      if (self.bytesPerIndex == 1) {
        uint8_t* pyrs = (uint8_t*)self.pyramids;
        i0 = pyrs[ID * 5 + 0];
        i1 = pyrs[ID * 5 + 1];
        i2 = pyrs[ID * 5 + 2];
        i3 = pyrs[ID * 5 + 3];
        i4 = pyrs[ID * 5 + 4];
      } else if (self.bytesPerIndex == 2) {
        uint16_t* pyrs = (uint16_t*)self.pyramids;
        i0 = pyrs[ID * 5 + 0];
        i1 = pyrs[ID * 5 + 1];
        i2 = pyrs[ID * 5 + 2];
        i3 = pyrs[ID * 5 + 3];
        i4 = pyrs[ID * 5 + 4];
      } else {
        uint32_t* pyrs = (uint32_t*)self.pyramids;
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

      if (interpolatePyramid(P,P0,P1,P2,P3,P4,S0,S1,S2,S3,S4,prd.dataValue)) {
        optixReportIntersection(0.f, 0);
        return;
      }
    }
  }

  OPTIX_INTERSECT_PROGRAM(WedgePointQuery)()
  {
    RayPayload &prd = owl::getPRD<RayPayload>();
    const auto &self = owl::getProgramData<UnstructuredElementData>();
    uint64_t primID = optixGetPrimitiveIndex() + self.offset; 
    float3 origin = optixGetObjectRayOrigin();
   
    // float maxima = self.maxima[self.numTetrahedra + self.numPyramids + primID];
    // if (maxima <= 0.f) return;

    vec3f P = {origin.x, origin.y, origin.z};
    
    // primID -= wedOffset;

    for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
      uint32_t ID = primID * ELEMENTS_PER_BOX + i;
      if (ID >= self.numWedges) return;

      uint64_t i0, i1, i2, i3, i4, i5;
      if (self.bytesPerIndex == 1) {
        uint8_t* wed = (uint8_t*)self.wedges;
        i0 = wed[ID * 6 + 0];
        i1 = wed[ID * 6 + 1];
        i2 = wed[ID * 6 + 2];
        i3 = wed[ID * 6 + 3];
        i4 = wed[ID * 6 + 4];
        i5 = wed[ID * 6 + 5];
      } else if (self.bytesPerIndex == 2) {
        uint16_t* wed = (uint16_t*)self.wedges;
        i0 = wed[ID * 6 + 0];
        i1 = wed[ID * 6 + 1];
        i2 = wed[ID * 6 + 2];
        i3 = wed[ID * 6 + 3];
        i4 = wed[ID * 6 + 4];
        i5 = wed[ID * 6 + 5];
      } else {
        uint32_t* wed = (uint32_t*)self.wedges;
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

      if (interpolateWedge(P,P0,P1,P2,P3,P4,P5,S0,S1,S2,S3,S4,S5,prd.dataValue)) {
        optixReportIntersection(0.f, 0);
        return;
      }
    }
  }

  OPTIX_INTERSECT_PROGRAM(HexahedraPointQuery)()
  {
    RayPayload &prd = owl::getPRD<RayPayload>();
    const auto &self = owl::getProgramData<UnstructuredElementData>();
    uint64_t primID = optixGetPrimitiveIndex() + self.offset; 
    float3 origin = optixGetObjectRayOrigin();
   
    // float maxima = self.maxima[self.numTetrahedra + self.numPyramids + self.numWedges + primID];
    // if (maxima <= 0.f) return;

    vec3f P = {origin.x, origin.y, origin.z};

    // primID -= hexOffset;
    for (int i = 0; i < ELEMENTS_PER_BOX; ++i) {
      uint32_t ID = primID * ELEMENTS_PER_BOX + i;
      if (ID >= self.numHexahedra) return;

      uint64_t i0, i1, i2, i3, i4, i5, i6, i7;
      if (self.bytesPerIndex == 1) {
        uint8_t* hexes = (uint8_t*)self.hexahedra;
        i0 = hexes[ID * 8 + 0];
        i1 = hexes[ID * 8 + 1];
        i2 = hexes[ID * 8 + 2];
        i3 = hexes[ID * 8 + 3];
        i4 = hexes[ID * 8 + 4];
        i5 = hexes[ID * 8 + 5];
        i6 = hexes[ID * 8 + 6];
        i7 = hexes[ID * 8 + 7];
      } else if (self.bytesPerIndex == 2) {
        uint16_t* hexes = (uint16_t*)self.hexahedra;
        i0 = hexes[ID * 8 + 0];
        i1 = hexes[ID * 8 + 1];
        i2 = hexes[ID * 8 + 2];
        i3 = hexes[ID * 8 + 3];
        i4 = hexes[ID * 8 + 4];
        i5 = hexes[ID * 8 + 5];
        i6 = hexes[ID * 8 + 6];
        i7 = hexes[ID * 8 + 7];
      } else {
        uint32_t* hexes = (uint32_t*)self.hexahedra;
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

      if (interpolateHexahedra(P,P0,P1,P2,P3,P4,P5,P6,P7,S0,S1,S2,S3,S4,S5,S6,S7,prd.dataValue)) {
        optixReportIntersection(0.f, 0);
        return;
      }
    }
  }

