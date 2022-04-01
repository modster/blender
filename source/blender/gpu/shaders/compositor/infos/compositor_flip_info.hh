/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(compositor_flip_shared)
    .local_group_size(16, 16)
    .sampler(0, ImageType::FLOAT_2D, "input_image")
    .image(0, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "output_image")
    .compute_source("compositor_flip.glsl");

GPU_SHADER_CREATE_INFO(compositor_flip_x)
    .additional_info("compositor_flip_shared")
    .define("FLIP_X")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(compositor_flip_y)
    .additional_info("compositor_flip_shared")
    .define("FLIP_Y")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(compositor_flip_x_and_y)
    .additional_info("compositor_flip_shared")
    .define("FLIP_X")
    .define("FLIP_Y")
    .do_static_compilation(true);
