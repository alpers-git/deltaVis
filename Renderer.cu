#include "Renderer.h"
// #include "helperMath.h"

#include <cub/cub.cuh>

#include "cuda_fp16.h"

// #include "hilbert.h"

#include <cuda_runtime.h>

#define CUDA_SYNC_CHECK()                                        \
    {                                                            \
        cudaDeviceSynchronize();                                 \
        cudaError_t rc = cudaGetLastError();                     \
        if (rc != cudaSuccess)                                   \
        {                                                        \
            fprintf(stderr, "error (%s: line %d): %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(rc)); \
            throw std::runtime_error("fatal cuda error");        \
        }                                                        \
    }

__global__ void _recalculateDensityRanges(
    int numPrims, bool is_background, owl::box4f *bboxes, // const uint8_t* nvdbData,
    cudaTextureObject_t texture, int numTexels,
    float2 volumeDomain, float2 xfDomain, float opacityScale,
    float *maxima)
{
    int primID = (blockIdx.x * blockDim.x + threadIdx.x);
    if (primID >= numPrims)
        return;

    // printf("primID %d level %d \n", primID, level);
    // if (level == -1)
    // printf("PrimID %d domain %f %f addr1 %f addr2 %f addrMin %d addrMax %d min %f max %f\n", primID, domain.x, domain.y, addr1, addr2, addrMin, addrMax,  minDensity, maxDensity);

    float mx = 0.f, mn = 0.f;
    if (!is_background)
    {
        mn = bboxes[primID].lower.w;
        mx = bboxes[primID].upper.w;
    }

    // empty box
    if (mx < mn)
    {
        maxima[primID] = 0.f;
        return;
    }

    // transform data min max to transfer function space
    float remappedMin1 = (mn - volumeDomain.x) / (volumeDomain.y - volumeDomain.x);
    float remappedMin = (remappedMin1 - xfDomain.x) / (xfDomain.y - xfDomain.x);
    float remappedMax1 = (mx - volumeDomain.x) / (volumeDomain.y - volumeDomain.x);
    float remappedMax = (remappedMax1 - xfDomain.x) / (xfDomain.y - xfDomain.x);
    float addr1 = remappedMin * numTexels;
    float addr2 = remappedMax * numTexels;

    int addrMin = min(max(int(min(floor(addr1), floor(addr2))), 0), numTexels - 1);
    int addrMax = min(max(int(max(ceil(addr1), ceil(addr2))), 0), numTexels - 1);

    // // When does this occur?
    // if (addrMin < 0) {
    //   maxima[primID] = 0.f;
    //   return;
    // }

    float maxDensity;
    for (int i = addrMin; i <= addrMax; ++i)
    {
        float density = tex2D<float4>(texture, float(i) / numTexels, 0.5f).w * opacityScale;
        if (i == addrMin)
            maxDensity = density;
        else
            maxDensity = max(maxDensity, density);
    }
    maxima[primID] = maxDensity;
}

namespace deltaVis
{

void Renderer::RecalculateDensityRanges()
{
    printf("Recalculating density ranges\n");
    float2 volumeDomain = {volDomain.lower, volDomain.upper};
    float2 tfnDomain = {0.0f, 1.0f};
    float opacityScale = this->opacityScale;
    cudaTextureObject_t colorMapTexture = this->colorMapTexture;
    int colorMapSize = this->colorMap.size();
    dim3 blockSize(32);
    uint32_t numThreads;
    bool isBackground;
    dim3 gridSize;
    owl::box4f *bboxes;
    float *maximaBuffer;
    {
        bboxes = (owl::box4f *)owlBufferGetPointer(gridBuffer , 0);
        isBackground = false;
        numThreads = macrocellDims.x * macrocellDims.y * macrocellDims.z;
        gridSize = dim3((numThreads + blockSize.x - 1) / blockSize.x);
        maximaBuffer = (float *)owlBufferGetPointer(majorantBuffer, 0);
        _recalculateDensityRanges<<<gridSize, blockSize>>>(
            numThreads, /*lvl*/ isBackground, bboxes,
            colorMapTexture, colorMapSize, volumeDomain, tfnDomain, opacityScale,
            maximaBuffer);
    }
    CUDA_SYNC_CHECK();
}

}