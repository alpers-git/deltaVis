#ifndef MACROCELL_BUILDER_HXX
#define MACROCELL_BUILDER_HXX

#include "owl/common.h"
#include "umesh/io/UMesh.h"
// Note, the above dependency can be found here: https://github.com/owl-project/owl

namespace deltaVis
{
    using namespace owl;

#ifdef __CUDACC__
#ifndef CUDA_DECORATOR
#define CUDA_DECORATOR __both__
#endif
#else
#ifndef CUDA_DECORATOR
#define CUDA_DECORATOR
#endif
#endif

    inline CUDA_DECORATOR box4f *BuildMacrocellGrid(vec3i dims,
                                                    umesh::vec3f *vertices,
                                                    float *scalars,
                                                    int numVertices)
    {
        // create a grid of macrocells dims.x * dims.y * dims.z find which vertices are in each macrocell
        // for each macrocell, find the min and max of the scalar values insert them as w values

        // allocate memory for the grid
        box4f *grid = new box4f[dims.x * dims.y * dims.z];
        // go over each vertex to find global bounds
        box4f globalBounds;

        // find the global bounds
        for (int i = 0; i < numVertices; i++)
        {
            globalBounds.extend(vec4f(vertices[i].x, vertices[i].y, vertices[i].z, scalars[i]));
        }
        vec4f cellSize = (globalBounds.upper - globalBounds.lower) / vec4f(dims.x, dims.y, dims.z, 1);

        // set the bounds for each macrocell
        int i, j, k;
        i = j = k = 0;
        for (float x = globalBounds.lower.x; x < globalBounds.upper.x; x += cellSize.x)
        {
            j = 0;
            for (float y = globalBounds.lower.y; y < globalBounds.upper.y; y += cellSize.y)
            {
                k = 0;
                for (float z = globalBounds.lower.z; z < globalBounds.upper.z; z += cellSize.z)
                {
                    float scalarMin = FLT_MAX;
                    float scalarMax = -FLT_MAX;
                    // find the min and max of the scalar values in the macrocell using the vertices
                    for (int l = 0; l < numVertices; l++)
                    {
                        if (vertices[l].x >= x && vertices[l].x < x + cellSize.x &&
                            vertices[l].y >= y && vertices[l].y < y + cellSize.y &&
                            vertices[l].z >= z && vertices[l].z < z + cellSize.z)
                        {
                            if (scalars[l] < scalarMin)
                                scalarMin = scalars[l];
                            if (scalars[l] > scalarMax)
                                scalarMax = scalars[l];
                        }
                    }
                    int idx = (j + k * dims.x) * dims.y + i;
                    if (idx >= dims.x * dims.y * dims.z)
                    {
                        printf("idx out of bounds: %d\n", idx);
                        continue;
                    }
                    // set the bounds for the macrocell
                    grid[idx] = box4f(
                        vec4f(x, y, z, scalarMin),
                        vec4f(x + cellSize.x, y + cellSize.y, z + cellSize.z, scalarMax));
                    k++;
                }
                j++;
            }
            i++;
        }
        return grid;
    }
}
#endif