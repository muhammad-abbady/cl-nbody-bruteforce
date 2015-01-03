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

//A kernel to calculate acceleration for the body
__kernel void accelerate(
//Bodies and their count
__global body *bodies, const unsigned int count,
//Constants
float G, float e
) {
    //get the global run id
    unsigned int i = get_global_id(0);
    
    //check if it's over the particles' count
    if (i >= count) return;
    
    //calculate acceleration
    bodies[i].ax = 0;
    bodies[i].ay = 0;
    unsigned int j = 0;
    float dx, dy, r, f;
    for (; j < count; ++j) {
        if (i == j) continue;
        dx = bodies[i].x - bodies[j].x;
        dy = bodies[i].y - bodies[j].y;
        r = max(sqrt(dx*dx + dy*dy), e);
        f = G * bodies[j].m / (r*r);
        
        bodies[i].ax -= f * (dx/r);
        bodies[i].ay -= f * (dy/r);
    }
}

//A kernel to integrate acceleration and velocity
__kernel void integrate(
//Bodies and their count
__global body *bodies, const unsigned int count,
//Constants
float dt, float decay
) {
    //get the global run id
    int i = get_global_id(0);
    
    //check if it's over the particles' count
    if (i >= count) return;
    
    //integrate
    bodies[i].vx += bodies[i].ax * dt;
    bodies[i].vy += bodies[i].ay * dt;
    bodies[i].vx *= decay;
    bodies[i].vy *= decay;
    bodies[i].x  += bodies[i].vx * dt;
    bodies[i].y  += bodies[i].vy * dt;
}