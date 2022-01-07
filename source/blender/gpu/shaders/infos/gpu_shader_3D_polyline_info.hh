
#include "gpu_interface_info.hh"
#include "gpu_shader_create_info.hh"

/* TODO(jbakker): Skipped as it needs a uniform/storage buffer. */
GPU_SHADER_CREATE_INFO(gpu_shader_3D_polyline_uniform_color)
    .vertex_source("gpu_shader_3D_polyline_vert.glsl")
    .geometry_source("gpu_shader_3D_polyline_geom.glsl")
    .fragment_source("gpu_shader_3D_polyline_frag.glsl")
    .do_static_compilation(true);
