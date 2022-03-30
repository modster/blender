/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2022 Blender Foundation.
 * All rights reserved.
 */

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
