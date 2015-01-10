//
//  kernels.cl
//  OpenCLNBody
//
//  Created by Muhammad Abbady on 1/3/15.
//  Copyright (c) 2015 Muhammad Abbady. All rights reserved.
//

struct _body {
    float x, y;
    float vx, vy;
    float ax, ay;
    float m;
};
typedef struct _body body;

void computeAcceleration(body* currentBody, float3 cachedBody, float G, float e) {
    float dx, dy, r, f;
    dx = currentBody->x - cachedBody.x;
    dy = currentBody->y - cachedBody.y;
    r = max(sqrt(dx*dx + dy*dy), e);
    f = G * cachedBody.z / (r*r);
    
    currentBody->ax -= f * (dx/r);
    currentBody->ay -= f * (dy/r);
}

//A kernel to calculate acceleration for the body
__kernel void accelerate(
                         //Bodies and their count
                         __global body *bodies, const unsigned int count, __local float3 *cache,
                         //Constants
                         float G, float e, float dt, float decay
                         ) {
    // get the global run id
    unsigned int i = get_global_id(0);
    unsigned int group_count = get_num_groups(0);
    unsigned int group_size = get_local_size(0);
    unsigned int local_id = get_local_id(0);
    
    // check if it's over the particles' count
    if (i >= count) return;
    
    // body cache
    body currentBody;
    currentBody.x = bodies[i].x;
    currentBody.y = bodies[i].y;
    currentBody.vx = bodies[i].vx;
    currentBody.vy = bodies[i].vy;
    currentBody.ax = 0;
    currentBody.ay = 0;
    
    // Loop on blocks
    for(unsigned int g = 0; g < group_count; ++g) {
        // cache a single body
        unsigned idx = g * group_size + local_id;
        cache[local_id].x = bodies[idx].x;
        cache[local_id].y = bodies[idx].y;
        cache[local_id].z = bodies[idx].m;
        
        // synchronize all work-items
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // calculate acceleration between the current body and the cache
        for(unsigned int j = 0; j < group_size;) {
            computeAcceleration(&currentBody, cache[j++], G, e);
            computeAcceleration(&currentBody, cache[j++], G, e);
            computeAcceleration(&currentBody, cache[j++], G, e);
            computeAcceleration(&currentBody, cache[j++], G, e);
            computeAcceleration(&currentBody, cache[j++], G, e);
            computeAcceleration(&currentBody, cache[j++], G, e);
            computeAcceleration(&currentBody, cache[j++], G, e);
            computeAcceleration(&currentBody, cache[j++], G, e);
        }
        
        // synchronize all work-items
        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    
    currentBody.vx += currentBody.ax * dt;
    currentBody.vy += currentBody.ay * dt;
    currentBody.x += currentBody.vx * dt;
    currentBody.y += currentBody.vy * dt;
    currentBody.vx *= decay;
    currentBody.vy *= decay;
    
    bodies[i].x = currentBody.x;
    bodies[i].y = currentBody.y;
    bodies[i].vx = currentBody.vx;
    bodies[i].vy = currentBody.vy;
}
