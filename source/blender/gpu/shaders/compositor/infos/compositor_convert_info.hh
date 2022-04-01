/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(compositor_convert_shared)
    .local_group_size(16, 16)
    .sampler(0, ImageType::FLOAT_2D, "input_sampler")
    .typedef_source("gpu_shader_compositor_type_conversion.glsl")
    .compute_source("compositor_convert.glsl");

GPU_SHADER_CREATE_INFO(compositor_convert_color_to_float)
    .additional_info("compositor_convert_shared")
    .image(0, GPU_R16F, Qualifier::WRITE, ImageType::FLOAT_2D, "output_image")
    .define("CONVERT_EXPRESSION", "vec4(float_from_vec4(texel))")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(compositor_convert_float_to_color)
    .additional_info("compositor_convert_shared")
    .image(0, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "output_image")
    .define("CONVERT_EXPRESSION", "vec4_from_float(texel.x)")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(compositor_convert_vector_to_color)
    .additional_info("compositor_convert_shared")
    .image(0, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "output_image")
    .define("CONVERT_EXPRESSION", "vec4_from_vec3(texel.xyz)")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(compositor_convert_color_to_half_color)
    .additional_info("compositor_convert_shared")
    .image(0, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "output_image")
    .define("CONVERT_EXPRESSION", "texel")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(compositor_convert_color_to_alpha)
    .additional_info("compositor_convert_shared")
    .image(0, GPU_R16F, Qualifier::WRITE, ImageType::FLOAT_2D, "output_image")
    .define("CONVERT_EXPRESSION", "texel.aaaa")
    .do_static_compilation(true);
