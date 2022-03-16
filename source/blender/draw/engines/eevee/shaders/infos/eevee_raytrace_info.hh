
/* FIXME(@fclem): This file is included inside the gpu module. We have to workaround to include
 * eevee header. */
#include "../../draw/engines/eevee/eevee_defines.hh"

#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Ray Generation
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_raytrace_raygen)
    .do_static_compilation(true)
    .local_group_size(RAYTRACE_GROUP_SIZE, RAYTRACE_GROUP_SIZE)
    .additional_info("eevee_shared")
    .storage_buf(0, Qualifier::READ_WRITE, "DispatchIndirectCommand", "dispatch_diffuse_buf")
    .storage_buf(1, Qualifier::READ_WRITE, "DispatchIndirectCommand", "dispatch_reflect_buf")
    .storage_buf(2, Qualifier::READ_WRITE, "DispatchIndirectCommand", "dispatch_refract_buf")
    .storage_buf(4, Qualifier::WRITE, "uint", "tiles_diffuse_buf[]")
    .storage_buf(5, Qualifier::WRITE, "uint", "tiles_reflect_buf[]")
    .storage_buf(6, Qualifier::WRITE, "uint", "tiles_refract_buf[]")
    .uniform_buf(3, "RaytraceBufferData", "raytrace_buffer_buf")
    .sampler(0, ImageType::FLOAT_2D, "gbuf_transmit_data_tx")
    .sampler(1, ImageType::FLOAT_2D, "gbuf_transmit_normal_tx")
    .sampler(2, ImageType::FLOAT_2D, "gbuf_reflection_normal_tx")
    .sampler(3, ImageType::FLOAT_2D, "depth_tx")
    .sampler(4, ImageType::UINT_2D, "stencil_tx")
    .image(0, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_ray_data_diffuse")
    .image(1, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_ray_data_reflect")
    .image(2, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_ray_data_refract")
    .compute_source("eevee_raytrace_raygen_comp.glsl")
    .additional_info("draw_view", "eevee_utility_texture", "eevee_sampling_data");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Screen Space Raytracing
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_raytrace_dispatch)
    .do_static_compilation(true)
    .local_group_size(1)
    .additional_info("eevee_shared")
    .storage_buf(0, Qualifier::READ_WRITE, "DispatchIndirectCommand", "dispatch_buf")
    .compute_source("eevee_raytrace_dispatch_comp.glsl");

GPU_SHADER_CREATE_INFO(eevee_raytrace_screen)
    .local_group_size(RAYTRACE_GROUP_SIZE, RAYTRACE_GROUP_SIZE)
    .additional_info("eevee_shared", "draw_view")
    .storage_buf(0, Qualifier::READ, "DispatchIndirectCommand", "dispatch_buf")
    .storage_buf(2, Qualifier::READ, "uint", "tiles_buf[]")
    .uniform_buf(4, "HiZData", "hiz_buf")
    .uniform_buf(5, "RaytraceBufferData", "raytrace_buffer_buf")
    .sampler(0, ImageType::FLOAT_2D, "radiance_tx")
    .sampler(1, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(2, ImageType::FLOAT_2D, "depth_tx")
    .image(0, GPU_RGBA16F, Qualifier::READ_WRITE, ImageType::FLOAT_2D, "inout_ray_data")
    .image(1, GPU_RGBA16F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_ray_radiance")
    .compute_source("eevee_raytrace_screen_comp.glsl")
    .additional_info("eevee_lightprobe_data", "eevee_sampling_data");

GPU_SHADER_CREATE_INFO(eevee_raytrace_screen_reflect)
    .do_static_compilation(true)
    .define("DO_REFLECTION", "true")
    .define("DO_REFRACTION", "false")
    .additional_info("eevee_raytrace_screen");

GPU_SHADER_CREATE_INFO(eevee_raytrace_screen_refract)
    .do_static_compilation(true)
    .define("DO_REFLECTION", "false")
    .define("DO_REFRACTION", "true")
    .additional_info("eevee_raytrace_screen");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Data
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_raytrace_data)
    .define("SCREEN_RAYTRACE")
    .uniform_buf(0, "RaytraceData", "raytrace_info_buf")
    .uniform_buf(3, "HiZData", "hiz_buf")
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(1, ImageType::FLOAT_2D, "radiance_tx");

/** \} */
