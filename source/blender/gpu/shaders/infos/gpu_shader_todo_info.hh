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

/** \file
 * \ingroup gpu
 */

#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_2D_image_multi_rect_color)
    .vertex_source("datatoc_gpu_shader_2D_image_multi_rect_vert.glsl")
    .fragment_source("datatoc_gpu_shader_image_varying_color_frag.glsl")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(gpu_shader_2D_widget_base)
    .vertex_source("gpu_shader_2D_widget_base_vert.glsl")
    .fragment_source("gpu_shader_2D_widget_base_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_WIDGET_BASE] =
        {
            .name = "GPU_SHADER_2D_WIDGET_BASE",
            .vert = datatoc_gpu_shader_2D_widget_base_vert_glsl,
            .frag = datatoc_gpu_shader_2D_widget_base_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_widget_base_inst)
    .vertex_source("gpu_shader_2D_widget_base_vert.glsl")
    .fragment_source("gpu_shader_2D_widget_base_frag.glsl")
    .define("USE_INSTANCE")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_WIDGET_BASE_INST] =
        {
            .name = "GPU_SHADER_2D_WIDGET_BASE_INST",
            .vert = datatoc_gpu_shader_2D_widget_base_vert_glsl,
            .frag = datatoc_gpu_shader_2D_widget_base_frag_glsl,
            .defs = "#define USE_INSTANCE\n",
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_widget_shadow)
    .vertex_source("gpu_shader_2D_widget_shadow_vert.glsl")
    .fragment_source("gpu_shader_2D_widget_shadow_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_WIDGET_SHADOW] =
        {
            .name = "GPU_SHADER_2D_WIDGET_SHADOW",
            .vert = datatoc_gpu_shader_2D_widget_shadow_vert_glsl,
            .frag = datatoc_gpu_shader_2D_widget_shadow_frag_glsl,
},
*/