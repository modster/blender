
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Surface Mesh Type
 * \{ */

GPU_SHADER_CREATE_INFO(draw_gpencil)
    .vertex_in(0, Type::IVEC4, "ma")
    .vertex_in(1, Type::IVEC4, "ma1")
    .vertex_in(2, Type::IVEC4, "ma2")
    .vertex_in(3, Type::IVEC4, "ma3")
    .vertex_in(4, Type::VEC4, "pos")
    .vertex_in(5, Type::VEC4, "pos1")
    .vertex_in(6, Type::VEC4, "pos2")
    .vertex_in(7, Type::VEC4, "pos3")
    .vertex_in(8, Type::VEC4, "uv1")
    .vertex_in(9, Type::VEC4, "uv2")
    .vertex_in(10, Type::VEC4, "col1")
    .vertex_in(11, Type::VEC4, "col2")
    .vertex_in(12, Type::VEC4, "fcol1");

GPU_SHADER_CREATE_INFO(eevee_surface_mesh)
    .vertex_in(0, Type::VEC3, "pos")
    .vertex_in(1, Type::VEC3, "nor");

GPU_SHADER_CREATE_INFO(eevee_surface_gpencil)
    .vertex_source("eevee_surface_gpencil_vert.glsl")
    .additional_info("draw_gpencil");

GPU_SHADER_CREATE_INFO(eevee_surface_hair)
    .vertex_source("eevee_surface_hair_vert.glsl")
    .additional_info("draw_hair");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Surface
 * \{ */

GPU_SHADER_CREATE_INTERFACE(eevee_surface_iface, "interp")
    .smooth(Type::VEC3, "P")
    .smooth(Type::VEC3, "N")
    .smooth(Type::VEC3, "hair_binormal")
    .smooth(Type::FLOAT, "hair_time")
    .smooth(Type::FLOAT, "hair_time_width")
    .smooth(Type::FLOAT, "hair_thickness")
    .flat(Type::INT, "hair_strand_id");

GPU_SHADER_CREATE_INFO(eevee_surface_deferred)
    .vertex_out(eevee_surface_iface)
    /* Diffuse or Transmission Color. */
    .fragment_out(0, Type::VEC3, "out_transmit_color")
    /* RG: Normal (negative if Tranmission), B: SSS ID, A: Min-Thickness */
    .fragment_out(1, Type::VEC4, "out_transmit_normal")
    /* RGB: SSS RGB Radius.
     * or
     * R: Transmission IOR, G: Transmission Roughness, B: Unused. */
    .fragment_out(2, Type::VEC3, "out_transmit_data")
    /* Reflection Color. */
    .fragment_out(3, Type::VEC3, "out_reflection_color")
    /* RG: Normal, B: Roughness X, A: Roughness Y. */
    .fragment_out(4, Type::VEC4, "out_reflection_normal")
    /* Volume Emission, Absorption, Scatter, Phase. */
    .fragment_out(5, Type::UVEC4, "out_volume_data")
    /* Emission. */
    .fragment_out(6, Type::VEC3, "out_emission_data")
    /* Transparent BSDF, Holdout. */
    .fragment_out(7, Type::VEC4, "out_transparency_data")
    .fragment_source("eevee_surface_deferred_frag.glsl")
    .additional_info("eevee_sampling_data", "eevee_utility_texture");

GPU_SHADER_CREATE_INFO(eevee_sampling_data).uniform_buf(0, "SamplingData", "sampling");

GPU_SHADER_CREATE_INFO(eevee_raytracing_data)
    .uniform_buf(0, "RaytraceData", "raytrace_diffuse")
    .uniform_buf(1, "RaytraceData", "raytrace_reflection")
    .uniform_buf(2, "RaytraceData", "raytrace_refraction")
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(1, ImageType::FLOAT_2D, "radiance_tx")
    .additional_info("eevee_sampling_data");

GPU_SHADER_CREATE_INFO(eevee_surface_forward)
    .uniform_buf(0, "HiZData", "hiz")
    .sampler(0, ImageType::FLOAT_2D, "hiz_tx")
    .sampler(1, ImageType::FLOAT_2D, "radiance_tx")
    .vertex_out(eevee_surface_iface)
    .fragment_out(0, Type::VEC4, "out_radiance", DualBlend::SRC_0)
    .fragment_out(0, Type::VEC4, "out_transmittance", DualBlend::SRC_1)
    .fragment_source("eevee_surface_forward_frag.glsl")
    .additional_info("eevee_transmittance_data",
                     "eevee_sampling_data",
                     "eevee_lightprobe_data",
                     "eevee_light_data",
                     "eevee_shadow_data");

GPU_SHADER_CREATE_INFO(eevee_surface_depth)
    .vertex_out(eevee_surface_iface)
    .fragment_source("eevee_surface_depth_frag.glsl")
    .additional_info("eevee_sampling_data");

GPU_SHADER_CREATE_INFO(eevee_surface_depth_simple)
    .vertex_out(eevee_surface_iface)
    .fragment_source("eevee_surface_depth_simple_frag.glsl");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Background
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_surface_background)
    .vertex_out(eevee_surface_iface)
    .fragment_out(0, Type::VEC4, "out_background")
    .fragment_source("eevee_surface_background_frag.glsl");

GPU_SHADER_CREATE_INFO(eevee_surface_world)
    .vertex_source("eevee_surface_world_vert.glsl")
    .additional_info("eevee_surface_background");

GPU_SHADER_CREATE_INFO(eevee_surface_lookdev)
    .vertex_in(0, Type::VEC3, "pos")
    .vertex_in(1, Type::VEC3, "nor")
    .vertex_source("eevee_surface_lookdev_vert.glsl")
    .additional_info("eevee_surface_background");

GPU_SHADER_CREATE_INFO(eevee_background_lookdev)
    .uniform_buf(0, "LightProbeInfoData", "probes_info")
    .sampler(0, ImageType::FLOAT_CUBE_ARRAY, "lightprobe_cube_tx")
    .push_constant(Type::FLOAT, "opacity")
    .push_constant(Type::FLOAT, "blur")
    .fragment_out(0, Type::VEC4, "out_background")
    .fragment_source("eevee_lookdev_background_frag.glsl")
    .additional_info("draw_fullscreen");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Volume
 * \{ */

GPU_SHADER_CREATE_INTERFACE(eevee_volume_iface, "interp")
    .smooth(Type::VEC3, "P_start")
    .smooth(Type::VEC3, "P_end");

GPU_SHADER_CREATE_INFO(eevee_volume_deferred)
    .sampler(0, ImageType::DEPTH_2D, "depth_max_tx")
    .vertex_in(0, Type::VEC3, "pos")
    .vertex_out(eevee_volume_iface)
    .fragment_out(0, Type::UVEC4, "out_volume_data")
    .fragment_out(1, Type::VEC4, "out_transparency_data")
    .typedef_source("eevee_shader_shared.hh")
    .vertex_source("eevee_volume_vert.glsl")
    .fragment_source("eevee_volume_deferred_frag.glsl")
    .additional_info("draw_fullscreen");

/** \} */
