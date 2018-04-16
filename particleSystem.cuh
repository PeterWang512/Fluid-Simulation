/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


    void setParameters(SimParams *hostParams);

    void integrateSystem(float *pos,
    					 float *oldPos,
                         float *vel,
                         float deltaTime,
                         uint numParticles);

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
                                     uint   numCells);

    void calcLambda(
    		float *lambda,
    		float *sortedPos,
    		uint  *cellStart,
    		uint  *cellEnd,
    		uint   numParticles);


    void calcDeltaP(
    		float *lambda,
    		float *delta_p,
    		float *sortedPos,
    		uint  *cellStart,
    		uint  *cellEnd,
    		uint   numParticles);


    void collide(float delta_t,
    		     float *newVel,
                 float *sortedPos,
                 float *sortedVel,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells);


    void update_position(float *sortedPos, float *delta_p, uint numParticles);


    void update_velocity(float delta_t,
    		             float *oldPos,
    		             float *sortedPos,
    		             float *vel,
    		             uint  *gridParticleIndex,    // input: sorted particle indices
    		             uint   numParticles);


    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

}
