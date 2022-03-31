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

GPU_SHADER_CREATE_INFO(compositor_split_viewer_shared)
    .local_group_size(16, 16)
    .push_constant(Type::FLOAT, "split_ratio")
    .push_constant(Type::IVEC2, "view_size")
    .sampler(0, ImageType::FLOAT_2D, "first_image")
    .sampler(1, ImageType::FLOAT_2D, "second_image")
    .image(0, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "output_image")
    .compute_source("compositor_split_viewer.glsl");

GPU_SHADER_CREATE_INFO(compositor_split_viewer_horizontal)
    .additional_info("compositor_split_viewer_shared")
    .define("SPLIT_HORIZONTAL")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(compositor_split_viewer_vertical)
    .additional_info("compositor_split_viewer_shared")
    .define("SPLIT_VERTICAL")
    .do_static_compilation(true);
