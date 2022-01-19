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

#include "gpu_interface_info.hh"
#include "gpu_shader_create_info.hh"

/* Interfaces between vertex stage and geometry stage. */
GPU_SHADER_INTERFACE_INFO(polyline_color_iface, "geom_in").smooth(Type::VEC4, "finalColor_g");
GPU_SHADER_INTERFACE_INFO(polyline_color_clipped_iface, "geom_in")
    .smooth(Type::VEC4, "finalColor_g")
    .smooth(Type::FLOAT, "clip_g");
GPU_SHADER_INTERFACE_INFO(polyline_uniform_color_clipped_iface, "geom_in")
    .smooth(Type::FLOAT, "clip_g");

/* Interfaces between geometry stage and fragment stage. */
GPU_SHADER_INTERFACE_INFO(polyline_clipped_iface, "geom_out")
    .smooth(Type::VEC4, "finalColor")
    .no_perspective(Type::FLOAT, "smoothline")
    .smooth(Type::FLOAT, "clip");
GPU_SHADER_INTERFACE_INFO(polyline_unclipped_iface, "geom_out")
    .smooth(Type::VEC4, "finalColor")
    .no_perspective(Type::FLOAT, "smoothline");

/* Abstract create info for polyline shaders. */
GPU_SHADER_CREATE_INFO(gpu_shader_3D_polyline)
    .vertex_in(0, Type::VEC3, "pos")
    .geometry_layout(PrimitiveIn::LINES, PrimitiveOut::TRIANGLE_STRIP, 4)
    .fragment_out(0, Type::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    /* Reserved space for Vec4 (16-19) to store uniform color. */
    .push_constant(20, Type::VEC2, "viewportSize")
    .push_constant(24, Type::FLOAT, "lineWidth")
    .push_constant(25, Type::BOOL, "lineSmooth")
    .typedef_source("GPU_shader_shared.h")
    .vertex_source("gpu_shader_3D_polyline_vert.glsl")
    .geometry_source("gpu_shader_3D_polyline_geom.glsl")
    .fragment_source("gpu_shader_3D_polyline_frag.glsl")
    .additional_info("gpu_srgb_to_framebuffer_space");

GPU_SHADER_CREATE_INFO(gpu_shader_3D_polyline_flat_color)
    .vertex_in(1, Type::VEC4, "color")
    .vertex_out(polyline_color_iface)
    .geometry_out(polyline_unclipped_iface)
    .define("FLAT")
    .additional_info("gpu_shader_3D_polyline")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(gpu_shader_3D_polyline_uniform_color)
    .geometry_out(polyline_unclipped_iface)
    .push_constant(16, Type::VEC4, "color")
    .define("UNIFORM")
    .additional_info("gpu_shader_3D_polyline")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(gpu_shader_3D_polyline_clipped_uniform_color)
    .vertex_out(polyline_uniform_color_clipped_iface)
    .geometry_out(polyline_clipped_iface)
    .uniform_buf(0, "ClippingData", "clipping_data")
    .push_constant(16, Type::VEC4, "color")
    .define("UNIFORM")
    .define("CLIP")
    .additional_info("gpu_shader_3D_polyline")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(gpu_shader_3D_polyline_smooth_color)
    .vertex_in(1, Type::VEC4, "color")
    .vertex_out(polyline_color_iface)
    .geometry_out(polyline_unclipped_iface)
    .define("SMOOTH")
    .additional_info("gpu_shader_3D_polyline")
    .do_static_compilation(true);
