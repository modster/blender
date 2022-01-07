#pragma once

#include "gpu_shader_create_info.hh"

GPU_SHADER_INTERFACE_INFO(flat_color_iface, "").flat(Type::VEC4, "finalColor");
GPU_SHADER_INTERFACE_INFO(smooth_color_iface, "").smooth(Type::VEC4, "finalColor");
GPU_SHADER_INTERFACE_INFO(smooth_tex_coord_interp_iface, "").smooth(Type::VEC2, "texCoord_interp");
