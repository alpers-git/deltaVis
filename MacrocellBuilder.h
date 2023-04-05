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
        // globalBounds.lower -= vec4f(0.01, 0.01, 0.01, 0.0f);
        // globalBounds.upper += vec4f(0.01, 0.01, 0.01, 0.0f);
        vec4f cellSize = (globalBounds.upper - globalBounds.lower) / vec4f(dims.x, dims.y, dims.z, 1);
        for (size_t i = 0; i < dims.x; i++)
        {
            for (size_t j = 0; j < dims.y; j++)
            {
                for (size_t k = 0; k < dims.z; k++)
                {
                    // find the bounds of the current cell
                    vec4f lower = globalBounds.lower + vec4f(i, j, k, INFINITY) * cellSize;
                    vec4f upper = lower + cellSize;
                    upper.w = -INFINITY;
                    box4f cellBounds(lower, upper);
                    // find the vertices that are in the current cell
                    for (int v = 0; v < numVertices; v++)
                    {
                        box3f tempBox(vec3f(lower.x, lower.y, lower.z), vec3f(upper.x, upper.y, upper.z));
                        if (tempBox.contains(vec3f(vertices[v].x, vertices[v].y, vertices[v].z)))
                        {
                            // extend the cell bounds to include the current vertex
                            cellBounds.extend(vec4f(vertices[v].x, vertices[v].y, vertices[v].z, scalars[v]));
                        }
                    }
                    // insert the cell bounds into the grid
                    grid[i + j * dims.x + k * dims.x * dims.y] = cellBounds;
                }
            }  
        }
        
        return grid;
    }
}
#endif