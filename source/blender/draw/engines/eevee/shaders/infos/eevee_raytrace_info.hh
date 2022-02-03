
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Variations
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_raytrace_diffuse).define("DIFFUSE");
GPU_SHADER_CREATE_INFO(eevee_raytrace_reflection).define("REFLECTION");
GPU_SHADER_CREATE_INFO(eevee_raytrace_refraction).define("REFRACTION");
GPU_SHADER_CREATE_INFO(eevee_raytrace_fallback).define("SKIP_TRACE");

#define EEVEE_RAYTRACE_FINAL_VARIATION(name, ...) \
  GPU_SHADER_CREATE_INFO(name).additional_info(__VA_ARGS__).do_static_compilation(true);

#define EEVEE_RAYTRACE_BSDF_VARIATIONS(prefix, ...) \
  EEVEE_RAYTRACE_FINAL_VARIATION(prefix##_diffuse, "eevee_raytrace_diffuse", __VA_ARGS__) \
  EEVEE_RAYTRACE_FINAL_VARIATION(prefix##_reflection, "eevee_raytrace_reflection", __VA_ARGS__) \
  EEVEE_RAYTRACE_FINAL_VARIATION(prefix##_refraction, "eevee_raytrace_refraction", __VA_ARGS__)

#define EEVEE_RAYTRACE_SKIP_VARIATIONS(prefix, ...) \
  EEVEE_RAYTRACE_BSDF_VARIATIONS(prefix##_fallback, "eevee_raytrace_fallback", __VA_ARGS__) \
  EEVEE_RAYTRACE_BSDF_VARIATIONS(prefix, __VA_ARGS__)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Raytracing
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_raytrace_raygen)
    .additional_info("eevee_shared")
    .uniform_buf(0, "RaytraceData", "raytrace_buf")
    .uniform_buf(1, "HiZData", "hiz_buf")
    .uniform_buf(2, "CubemapData", "cubes_buf[CULLING_ITEM_BATCH]")
    .uniform_buf(3, "LightProbeInfoData", "probes_buf")
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(1, ImageType::FLOAT_2D, "hiz_front_tx")
    .sampler(2, ImageType::FLOAT_CUBE_ARRAY, "lightprobe_cube_tx")
    .sampler(3, ImageType::FLOAT_2D, "radiance_tx")
    .sampler(4, ImageType::FLOAT_2D, "combined_tx")
    .sampler(5, ImageType::FLOAT_2D, "cl_color_tx")
    .sampler(6, ImageType::FLOAT_2D, "cl_normal_tx")
    .sampler(7, ImageType::FLOAT_2D, "cl_data_tx")
    .fragment_out(0, Type::VEC4, "out_ray_data")
    .fragment_out(1, Type::VEC4, "out_ray_radiance")
    .fragment_source("eevee_raytrace_raygen_frag.glsl")
    .additional_info("draw_fullscreen", "eevee_utility_texture", "eevee_sampling_data");

EEVEE_RAYTRACE_SKIP_VARIATIONS(eevee_raytrace_raygen, "eevee_raytrace_raygen");

GPU_SHADER_CREATE_INFO(eevee_raytrace_denoise)
    .additional_info("eevee_shared")
    .local_group_size(8, 8)
    .uniform_buf(0, "RaytraceData", "raytrace_buf")
    .uniform_buf(1, "HiZData", "hiz_buf")
    .uniform_buf(2, "RaytraceBufferData", "rtbuf_buf")
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(1, ImageType::FLOAT_2D, "ray_data_tx")
    .sampler(2, ImageType::FLOAT_2D, "ray_radiance_tx")
    .sampler(3, ImageType::FLOAT_2D, "cl_color_tx")
    .sampler(4, ImageType::FLOAT_2D, "cl_normal_tx")
    .sampler(5, ImageType::FLOAT_2D, "cl_data_tx")
    .sampler(6, ImageType::FLOAT_2D, "ray_history_tx")
    .sampler(7, ImageType::FLOAT_2D, "ray_variance_tx")
    .image(0, GPU_RGBA16F, Qualifier::READ_WRITE, ImageType::FLOAT_2D, "out_history_img")
    .image(1, GPU_R8, Qualifier::READ_WRITE, ImageType::FLOAT_2D, "out_variance_img")
    .compute_source("eevee_raytrace_denoise_comp.glsl");

EEVEE_RAYTRACE_BSDF_VARIATIONS(eevee_raytrace_denoise, "eevee_raytrace_denoise");

GPU_SHADER_CREATE_INFO(eevee_raytrace_resolve)
    .additional_info("eevee_shared")
    .uniform_buf(1, "HiZData", "hiz_buf")
    .sampler(2, ImageType::FLOAT_2D, "ray_radiance_tx")
    .sampler(3, ImageType::FLOAT_2D, "cl_color_tx")
    .sampler(4, ImageType::FLOAT_2D, "cl_normal_tx")
    .sampler(5, ImageType::FLOAT_2D, "cl_data_tx")
    .sampler(7, ImageType::FLOAT_2D, "ray_variance_tx")
    .fragment_out(0, Type::VEC4, "out_combined")
    .fragment_out(1, Type::VEC4, "out_diffuse")
    .fragment_out(2, Type::VEC3, "out_specular")
    .fragment_source("eevee_raytrace_resolve_frag.glsl")
    .additional_info("draw_fullscreen");

EEVEE_RAYTRACE_BSDF_VARIATIONS(eevee_raytrace_resolve, "eevee_raytrace_resolve");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Data
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_raytrace_data)
    .uniform_buf(0, "RaytraceData", "raytrace_diffuse")
    .uniform_buf(1, "RaytraceData", "raytrace_reflection")
    .uniform_buf(2, "RaytraceData", "raytrace_refraction")
    .uniform_buf(3, "HiZData", "hiz_buf")
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(1, ImageType::FLOAT_2D, "radiance_tx");

/** \} */
