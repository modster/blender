/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(compositor_blur_shared)
    .local_group_size(16, 16)
    .sampler(0, ImageType::FLOAT_2D, "input_image")
    .sampler(1, ImageType::FLOAT_1D, "weights")
    .image(0, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "output_image")
    .compute_source("compositor_blur.glsl");

GPU_SHADER_CREATE_INFO(compositor_blur_horizontal)
    .additional_info("compositor_blur_shared")
    .define("BLUR_HORIZONTAL")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(compositor_blur_vertical)
    .additional_info("compositor_blur_shared")
    .define("BLUR_VERTICAL")
    .do_static_compilation(true);
