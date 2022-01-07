
#include "gpu_interface_info.hh"
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_3D_depth_only)
    .vertex_in(0, Type::VEC3, "pos")
    .vertex_out(flat_color_iface)
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .vertex_source("gpu_shader_3D_vert.glsl")
    .fragment_source("gpu_shader_depth_only_frag.glsl")
    .do_static_compilation(true);
