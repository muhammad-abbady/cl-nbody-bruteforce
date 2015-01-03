//
//  main.cpp
//  OpenCLNBody
//
//  Created by Muhammad Abbady on 1/2/15.
//  Copyright (c) 2015 Muhammad Abbady. All rights reserved.
//

#include <iostream>
#include <math.h>
#include <string>
#include <fstream>

#ifdef __APPLE__
#include <OpenCL/OpenCL.h>
#include <GLUT/GLUT.h>
#endif

// the structure for the bodies
struct _body {
    float x, y;
    float vx, vy;
    float ax, ay;
    float m;
};
typedef struct _body body;

// Data
body *bodies;

// Constants
const float G = 6.67384e-11;
const float e = 1;
const float PI = acosf(-1);
const unsigned int number_of_particles = 1 << 10;

// Global Variables
float dt = 10000;
float decay = 1;

// GL global variables
GLuint bodies_vbo;

// CL global variables
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel acceleration_kernel;
cl_kernel integration_kernel;
cl_mem bodies_cl_memory;

// Window global variables
unsigned int width = 1000;
unsigned int height = 600;
int current_mouse_x = 0;
int current_mouse_y = 0;

// To calculate the size of bodies in bytes
#define SIZE_OF_BODIES sizeof(body) * number_of_particles

// Helper functions for CL
void runIteration(float dt);
void updateHostMemoryFromCLBuffer();

// Helper functions for GL
void render();

// The GLUT update function
void update(int);

// GLUT callbacks
void mouseMove(int, int);
void resize(int, int);

// generate random floats from 0.0 to 1.0 inclusive
float frand() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

// Prints an error and exits
void error(std::string message) {
    std::cout << message << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char ** argv) {
    int err = 0;
    
    // Initialize bodies
    //--------------------------------------------------------------------------
    bodies = new body[number_of_particles];
    float maximumMass = 0.1;
    float maximumRadius = 500;
    float minimumRadius = 2;
    
    // Set the first particle's mass to be the heaviest, and make it centered
    // (like the center of a galaxy)
    bodies[0].m = maximumMass * 10000;
    bodies[0].x = 0;
    bodies[0].y = 0;
    bodies[0].vx = 0;
    bodies[0].vy = 0;
    bodies[0].ax = 0;
    bodies[0].ay = 0;
    
    // The rest of the particles will have small masses, and they should rotate
    // around the center.
    float radius, angle, dx, dy, r, ve;
    for (unsigned int i = 1; i < number_of_particles; i++) {
        //  Initialize the position of the particle (I used a random radius and
        //  angle to shape the position into a circle).
        radius = minimumRadius + frand() * (maximumRadius - minimumRadius);
        angle = frand() * PI * 2;
        bodies[i].x = radius * cos(angle);
        bodies[i].y = radius * sin(angle);
        
        // Calculate the distance between the particle and the center particle.
        dx = bodies[i].x - bodies[0].x;
        dy = bodies[i].y - bodies[0].y;
        r = sqrt(dx*dx + dy*dy);
        
        // Calculate the orbiting velocity
        // (Check http:// en.wikipedia.org/wiki/Orbital_speed/ ).
        ve = sqrt(G * bodies[0].m / r);
        
        // Set the speed of the particle
        bodies[i].vx = ve * dy / r;
        bodies[i].vy = ve * -dx / r;
        
        // Set the mass of the particle.
        bodies[i].m = frand() * maximumMass;
        
        // Set the acceleration with zero (will be used later).
        bodies[i].ax = 0;
        bodies[i].ay = 0;
    }
    //--------------------------------------------------------------------------
    
    
    // Initializing OpenGL/GLUT
    //--------------------------------------------------------------------------
    // Glut setup
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("N-Body Simulation");
    glutDisplayFunc(render);
    glutPassiveMotionFunc(mouseMove);
    glutReshapeFunc(resize);
    
    // Creating a vertex buffer
    glGenBuffers(1, &bodies_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, bodies_vbo);
    glBufferData(GL_ARRAY_BUFFER, SIZE_OF_BODIES, bodies, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, NULL);
    
    // Enable smoothing
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glPointSize(0);
    
    // Enable alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //--------------------------------------------------------------------------
    
    
    // Initializing OpenCL
    //--------------------------------------------------------------------------
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err) {
        error("Error getting device ids");
    }
    
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    if (err) {
        error("Error creating context");
    }
    
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err) {
        error("Error creating a command queue");
    }
    
    std::ifstream source_file("kernels.cl");
    std::string source((std::istreambuf_iterator<char>(source_file)),
                       (std::istreambuf_iterator<char>()));
    const char *source_c_str = source.c_str();
    
    program = clCreateProgramWithSource(context, 1,
                                        (const char **)&source_c_str, 0, &err);
    if (err) {
        error("Error creating a program");
    }
    
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err) {
        size_t length;
        char buffer[2048];
        
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &length);
        error(buffer);
    }
    
    acceleration_kernel = clCreateKernel(program, "accelerate", &err);
    if (err) {
        error("Error creating the acceleration kernel");
    }
    
    integration_kernel = clCreateKernel(program, "integrate", &err);
    if (err) {
        error("Error creating the integration kernel");
    }
    
    bodies_cl_memory = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      SIZE_OF_BODIES, NULL, &err);
    if (err) {
        error("Error creating the cl buffer");
    }
    
    err = clEnqueueWriteBuffer(queue, bodies_cl_memory, CL_TRUE, 0,
                               SIZE_OF_BODIES, bodies, 0, NULL, NULL);
    if (err) {
        error("Error uploading data to cl buffer");
    }
    
    // --------------------------------------------------------------------------
    
    
    // start the GLUT loop
    // --------------------------------------------------------------------------
    
    update(0);
    glutMainLoop();
    //--------------------------------------------------------------------------
    
    
    // release CL objects
    //--------------------------------------------------------------------------
    clReleaseKernel(integration_kernel);
    clReleaseKernel(acceleration_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    //--------------------------------------------------------------------------
    
    // release GL objects
    //--------------------------------------------------------------------------
    glDeleteBuffers(1, &bodies_vbo);
    //--------------------------------------------------------------------------
    
    // release allocated memory
    //--------------------------------------------------------------------------
    delete [] bodies;
    //--------------------------------------------------------------------------
    
    return 0;
}


void runIteration(float dt) {
    int err = 0;
    size_t global_size = 0;
    size_t workgroup_size = 0;
    
    // Update the position of the center particle to the current mouse position.
    bodies[0].x = current_mouse_x;
    bodies[0].y = current_mouse_y;
    
    err = clEnqueueWriteBuffer(queue, bodies_cl_memory, CL_TRUE, 0,
                               sizeof(float) * 2, bodies, 0, NULL, NULL);
    if (err) {
        error("Error setting the center particle position");
    }
    
    
    // Run the acceleration kernel
    // Setting the kernel arguments
    err |= clSetKernelArg(acceleration_kernel, 0,
                          sizeof(cl_mem), &bodies_cl_memory);
    err |= clSetKernelArg(acceleration_kernel, 1,
                          sizeof(unsigned int), &number_of_particles);
    err |= clSetKernelArg(acceleration_kernel, 2,
                          sizeof(float), &G);
    err |= clSetKernelArg(acceleration_kernel, 3,
                          sizeof(float), &e);
    if (err) {
        error("Failed to set acceleration kernel arguments");
    }
    
    // Get the maximum work group size for the kernel
    err = clGetKernelWorkGroupInfo(acceleration_kernel, device,
                                   CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(workgroup_size),
                                   &workgroup_size, NULL);
    if (err) {
        error("Error getting acceleration kernel work group size");
    }
    
    // Execute the kernel
    err = 0;
    global_size = workgroup_size * (number_of_particles / workgroup_size + 1);
        err |= clEnqueueNDRangeKernel(queue, acceleration_kernel, 1,
                                      NULL, &global_size,
                                      &workgroup_size, 0, NULL, NULL);
    
    if (err) {
        error("Error executing acceleration kernel");
    }
    
    // Wait for the kernel to finish
    clFinish(queue);
    
    // Run the integration kernel
    // Set the kernel arguments
    err = clSetKernelArg(integration_kernel, 0,
                         sizeof(cl_mem), &bodies_cl_memory);
    err |= clSetKernelArg(integration_kernel, 1,
                         sizeof(unsigned int), &number_of_particles);
    err |= clSetKernelArg(integration_kernel, 2,
                         sizeof(float), &dt);
    err |= clSetKernelArg(integration_kernel, 3,
                         sizeof(float), &decay);
    
    if (err) {
        error("Failed to set integration kernel arguments");
    }
    
    // Get the maximum work group size for the kernel
    err = clGetKernelWorkGroupInfo(integration_kernel, device,
                                   CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(workgroup_size),
                                   &workgroup_size, NULL);
    if (err) {
        error("Error getting integration kernel work group size");
    }
    
    // Execute the integration kernel.
    global_size = workgroup_size * (number_of_particles / workgroup_size + 1);
    err = clEnqueueNDRangeKernel(queue, integration_kernel, 1,
                                 NULL, &global_size, &workgroup_size,
                                 0, NULL, NULL);
    if (err) {
        error("Error executing integration kernel");
    }
    
    // Wait for the kernel to finish
    clFinish(queue);
    
    // Update the host memory
    err = clEnqueueReadBuffer(queue, bodies_cl_memory, CL_TRUE, 0,
                              SIZE_OF_BODIES, bodies, 0, NULL, NULL);
    if (err) {
        error("Error reading cl buffer");
    }
}

void render() {
    // Clearing the buffer with black
    glClearColor(0.0, 0.0, 0.0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Adjusting the view
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-0.5 * width, 0.5 * width, -0.5 * height, 0.5 * height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Set a color for particles
    glColor4f(0.5, 0.7, 1.0, 0.4);
    
    // Update the vertex buffer and draw
    glBindBuffer(GL_ARRAY_BUFFER, bodies_vbo);
    glBufferData(GL_ARRAY_BUFFER, SIZE_OF_BODIES, bodies, GL_STATIC_DRAW);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, sizeof(body), NULL);
    glDrawArrays(GL_POINTS, 0, number_of_particles);
    
    // Refresh the view
    glFlush();
    glutSwapBuffers();
}

void update(int) {
    // Run CL iteration
    runIteration(dt);
    
    // Render
    glutPostRedisplay();
    
    // Call the update function with no delay
    glutTimerFunc(0, update, 0);
}

void resize(int w, int h) {
    width = w;
    height = h;
}

void mouseMove(int x, int y) {
    current_mouse_x = x - width / 2;
    current_mouse_y = -y + height / 2;
}