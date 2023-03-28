# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid FEM
#
# Shows how to set up a rigid sphere colliding with an FEM beam
# using wp.sim.ModelBuilder().
#
###########################################################################

import os
import math

import numpy as np
import torch

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

default_tri_ke = 100.0
default_tri_ka = 100.0
default_tri_kd = 10.0
default_tri_drag = 0.0
default_tri_lift = 0.0

# Default edge bending properties
default_edge_ke = 100.0
default_edge_kd = 0.0

def add_soft_grid(builder,
                  points,
                  tets,
                  vel,
                  density: float,
                  k_mu: float,
                  k_lambda: float,
                  k_damp: float,
                  tri_ke: float=default_tri_ke,
                  tri_ka: float=default_tri_ka,
                  tri_kd: float=default_tri_kd,
                  tri_drag: float=default_tri_drag,
                  tri_lift: float=default_tri_lift):

    start_vertex = len(builder.particle_q)

    for i in range(points.shape[0]):
        builder.add_particle(points[i], vel, density)

    # dict of open faces
    faces = {}

    def add_face(i: int, j: int, k: int):
        key = tuple(sorted((i, j, k)))

        if key not in faces:
            faces[key] = (i, j, k)
        else:
            del faces[key]

    def add_tet(i: int, j: int, k: int, l: int):
        builder.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

        add_face(i, k, j)
        add_face(j, k, l)
        add_face(i, j, l)
        add_face(i, l, k)

    def grid_index(x, y, z):
        return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

    for i in range(len(tets)):
        tet = tets[i]
        add_tet(tet[0], tet[1], tet[2], tet[3])
    # add triangles
    for k, v in faces.items():
        builder.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)


class LayerStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, qd, env, frame):

        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        ctx.tape = wp.Tape()
        ctx.env = env        
        # allocate output
        ctx.state_0 = env.model.state()
        ctx.state_0.particle_q = wp.from_torch(q, dtype=wp.vec3)
        ctx.state_0.particle_qd = wp.from_torch(qd, dtype=wp.vec3)

        ctx.state_1 = env.model.state()

        ctx.frame = frame
        ctx.state_0.clear_forces()
        ctx.state_1.clear_forces()


        with ctx.tape:
            env.integrator.simulate(env.model, ctx.state_0, ctx.state_1, env.sim_dt)



        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        return (wp.to_torch(ctx.state_1.particle_q),
                wp.to_torch(ctx.state_1.particle_qd))



    @staticmethod
    def backward(ctx, adj_q, adj_qd):

        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        # map incoming Torch grads to our output variables
        ctx.state_1.particle_q.grad = wp.from_torch(adj_q, dtype=wp.vec3)
        ctx.state_1.particle_qd.grad = wp.from_torch(adj_qd, dtype=wp.vec3)

        ctx.tape.backward()

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        if ctx.state_0.particle_q in ctx.tape.gradients.keys() and ctx.state_0.particle_qd in ctx.tape.gradients.keys():
            grad_q = wp.to_torch(ctx.tape.gradients[ctx.state_0.particle_q])
            grad_qd = wp.to_torch(ctx.tape.gradients[ctx.state_0.particle_qd])
            # print("grad finish", ctx.frame)
        else:
            grad_q = torch.zeros_like(adj_q)
            grad_qd = torch.zeros_like(adj_q)
            # print("no grad", ctx.frame)
        # return adjoint w.r.t. inputs
        return (grad_q,
                grad_qd,
                None, None)


class Ball3Gravity:

    def __init__(self, points_ref, eles_ref):
        stage = "example_Ball3Gravity.usd"
        self.sim_width = 8
        self.sim_height = 8

        self.sim_fps = 60.0
        self.sim_substeps = 32
        self.sim_duration = 5.0
        self.sim_frames = int(self.sim_duration*self.sim_fps)
        self.sim_dt = (1.0/self.sim_fps)/self.sim_substeps
        self.sim_time = 0.0
        self.sim_iterations = 1
        self.sim_relaxation = 1.0

        builder = wp.sim.ModelBuilder()

        add_soft_grid(
            builder,
            points_ref,
            eles_ref,
            vel=(0.0, 0.0, 0.0), 
            density=1.0, 
            k_mu=50000.0, 
            k_lambda=20000.0,
            k_damp=0.0)

        self.model = builder.finalize()
        self.model.gravity = np.array((0.0, 0.0, 0.0))
        self.model.ground = False
        self.model.soft_contact_distance = 0.01
        self.model.soft_contact_ke = 1.e+3
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_kf = 1.e+3

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # self.model.collide(self.state_0)
        # self.renderer = wp.sim.render.SimRenderer(self.model, stage)

        self.layer_step = LayerStep.apply

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            for s in range(self.sim_substeps):

                wp.sim.collide(self.model, self.state_0)

                self.state_0.clear_forces()
                self.state_1.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.sim_time += self.sim_dt

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True): 
            time = 0.0 if is_live else self.sim_time
            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

