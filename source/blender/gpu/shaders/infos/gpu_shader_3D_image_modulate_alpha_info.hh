
#include "gpu_interface_info.hh"
#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_3D_image_modulate_alpha)
    .vertex_in(0, Type::VEC3, "pos")
    .vertex_in(1, Type::VEC2, "texCoord")
    .vertex_out(smooth_tex_coord_interp_iface)
    .fragment_out(0, Type::VEC4, "fragColor")
    .push_constant(0, Type::MAT4, "ModelViewProjectionMatrix")
    .push_constant(16, Type::FLOAT, "alpha")
    .sampler(0, ImageType::FLOAT_2D, "image", Frequency::PASS)
    .vertex_source("gpu_shader_3D_image_vert.glsl")
    .fragment_source("gpu_shader_image_modulate_alpha_frag.glsl")
    .do_static_compilation(true);
