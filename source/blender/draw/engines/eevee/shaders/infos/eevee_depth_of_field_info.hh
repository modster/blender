
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Setup
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_bokeh_lut)
    .do_static_compilation(true)
    .additional_info("eevee_shared")
    .uniform_buf(0, "DepthOfFieldData", "dof_buf")
    .fragment_out(0, Type::VEC2, "out_gather_lut")
    .fragment_out(1, Type::FLOAT, "out_scatter_lut")
    .fragment_out(2, Type::FLOAT, "out_resolve_lut")
    .fragment_source("eevee_depth_of_field_bokeh_lut_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_setup)
    .do_static_compilation(true)
    .additional_info("eevee_shared")
    .uniform_buf(0, "DepthOfFieldData", "dof_buf")
    .sampler(0, ImageType::FLOAT_2D, "color_tx", Frequency::PASS)
    .sampler(1, ImageType::DEPTH_2D, "depth_tx", Frequency::PASS)
    .fragment_out(0, Type::VEC4, "out_color")
    .fragment_out(1, Type::VEC2, "out_coc")
    .fragment_source("eevee_depth_of_field_setup_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_filter)
    .do_static_compilation(true)
    .additional_info("eevee_shared")
    .sampler(0, ImageType::FLOAT_2D, "color_tx", Frequency::PASS)
    .sampler(1, ImageType::FLOAT_2D, "weight_tx", Frequency::PASS)
    .fragment_out(0, Type::VEC4, "out_color")
    .fragment_out(1, Type::FLOAT, "out_weight")
    .fragment_source("eevee_depth_of_field_filter_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_reduce_copy)
    .do_static_compilation(true)
    .additional_info("eevee_shared")
    .uniform_buf(0, "DepthOfFieldData", "dof_buf")
    .sampler(0, ImageType::FLOAT_2D, "color_tx", Frequency::PASS)
    .sampler(1, ImageType::FLOAT_2D, "coc_tx", Frequency::PASS)
    .sampler(2, ImageType::FLOAT_2D, "downsampled_tx", Frequency::PASS)
    .fragment_out(0, Type::VEC4, "out_color_gather")
    .fragment_out(1, Type::FLOAT, "out_coc")
    .fragment_out(2, Type::VEC3, "out_color_scatter")
    .fragment_source("eevee_depth_of_field_reduce_copy_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_reduce_downsample)
    .do_static_compilation(true)
    .additional_info("eevee_shared")
    .sampler(0, ImageType::FLOAT_2D, "color_tx", Frequency::PASS)
    .sampler(1, ImageType::FLOAT_2D, "coc_tx", Frequency::PASS)
    .fragment_out(0, Type::VEC4, "out_color")
    .fragment_source("eevee_depth_of_field_reduce_downsample_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_reduce_recursive)
    .do_static_compilation(true)
    .additional_info("eevee_shared")
    .uniform_buf(0, "DepthOfFieldData", "dof_buf")
    .uniform_buf(1, "SamplingData", "sampling_buf")
    .sampler(0, ImageType::DEPTH_2D, "color_tx", Frequency::PASS)
    .sampler(1, ImageType::FLOAT_2D, "coc_tx", Frequency::PASS)
    .fragment_out(0, Type::VEC4, "out_color")
    .fragment_out(1, Type::FLOAT, "out_coc")
    .fragment_source("eevee_depth_of_field_reduce_recursive_frag.glsl")
    .additional_info("draw_fullscreen");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Variations
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_no_lut)
    .define("DOF_BOKEH_TEXTURE", "false")
    /**
     * WORKAROUND(@fclem): This is to keep the code as is for now. The bokeh_lut_tx is referenced
     * even if not used after optimisation. But we don't want to include it in the create infos.
     */
    .define("bokeh_lut_tx", "color_tx");

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_lut)
    .define("DOF_BOKEH_TEXTURE", "true")
    .sampler(5, ImageType::FLOAT_2D, "bokeh_lut_tx", Frequency::PASS);

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_background).define("DOF_FOREGROUND_PASS", "false");
GPU_SHADER_CREATE_INFO(eevee_depth_of_field_foreground).define("DOF_FOREGROUND_PASS", "true");
GPU_SHADER_CREATE_INFO(eevee_depth_of_field_hq).define("DOF_SLIGHT_FOCUS_DENSITY", "4");
GPU_SHADER_CREATE_INFO(eevee_depth_of_field_lq).define("DOF_SLIGHT_FOCUS_DENSITY", "2");

#define EEVEE_DOF_FINAL_VARIATION(name, ...) \
  GPU_SHADER_CREATE_INFO(name).additional_info(__VA_ARGS__).do_static_compilation(true);

#define EEVEE_DOF_LUT_VARIATIONS(prefix, ...) \
  EEVEE_DOF_FINAL_VARIATION(prefix##_lut, "eevee_depth_of_field_lut", __VA_ARGS__) \
  EEVEE_DOF_FINAL_VARIATION(prefix, "eevee_depth_of_field_no_lut", __VA_ARGS__)

#define EEVEE_DOF_GROUND_VARIATIONS(name, ...) \
  EEVEE_DOF_LUT_VARIATIONS(name##_background, "eevee_depth_of_field_background", __VA_ARGS__) \
  EEVEE_DOF_LUT_VARIATIONS(name##_foreground, "eevee_depth_of_field_foreground", __VA_ARGS__)

#define EEVEE_DOF_HQ_VARIATIONS(name, ...) \
  EEVEE_DOF_LUT_VARIATIONS(name##_hq, "eevee_depth_of_field_hq", __VA_ARGS__) \
  EEVEE_DOF_LUT_VARIATIONS(name##_lq, "eevee_depth_of_field_lq", __VA_ARGS__)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Gather
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_gather)
    .additional_info("eevee_shared")
    .uniform_buf(0, "DepthOfFieldData", "dof_buf")
    .uniform_buf(1, "SamplingData", "sampling_buf")
    .sampler(0, ImageType::FLOAT_2D, "color_tx", Frequency::PASS)
    .sampler(1, ImageType::FLOAT_2D, "color_bilinear_tx", Frequency::PASS)
    .sampler(2, ImageType::FLOAT_2D, "coc_tx", Frequency::PASS)
    .sampler(3, ImageType::FLOAT_2D, "tiles_fg_tx", Frequency::PASS)
    .sampler(4, ImageType::FLOAT_2D, "tiles_bg_tx", Frequency::PASS)
    .fragment_out(0, Type::VEC4, "out_color")
    .fragment_out(1, Type::FLOAT, "out_weight")
    .fragment_out(2, Type::VEC2, "out_occlusion")
    .fragment_source("eevee_depth_of_field_gather_frag.glsl")
    .additional_info("draw_fullscreen");

EEVEE_DOF_GROUND_VARIATIONS(eevee_depth_of_field_gather, "eevee_depth_of_field_gather")

/** \} */

/* -------------------------------------------------------------------- */
/** \name Gather Holefill
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_gather_holefill)
    .additional_info("eevee_shared")
    .uniform_buf(0, "DepthOfFieldData", "dof_buf")
    .uniform_buf(1, "SamplingData", "sampling_buf")
    .sampler(0, ImageType::FLOAT_2D, "color_tx", Frequency::PASS)
    .sampler(1, ImageType::FLOAT_2D, "color_bilinear_tx", Frequency::PASS)
    .sampler(2, ImageType::FLOAT_2D, "coc_tx", Frequency::PASS)
    .sampler(3, ImageType::FLOAT_2D, "tiles_fg_tx", Frequency::PASS)
    .sampler(4, ImageType::FLOAT_2D, "tiles_bg_tx", Frequency::PASS)
    .fragment_out(0, Type::VEC4, "out_color")
    .fragment_out(1, Type::FLOAT, "out_weight")
    .fragment_source("eevee_depth_of_field_gather_holefill_frag.glsl")
    .additional_info("draw_fullscreen", "eevee_depth_of_field_no_lut");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Scatter
 * \{ */

GPU_SHADER_INTERFACE_INFO(eevee_depth_of_field_scatter_iface, "")
    /** Colors, weights, and Circle of confusion radii for the 4 pixels to scatter. */
    .flat(Type::VEC4, "color1")
    .flat(Type::VEC4, "color2")
    .flat(Type::VEC4, "color3")
    .flat(Type::VEC4, "color4")
    .flat(Type::VEC4, "weights")
    .flat(Type::VEC4, "cocs")
    /** Sprite center position. In pixels. */
    .flat(Type::VEC2, "spritepos")
    /** MaxCoC. */
    .flat(Type::FLOAT, "spritesize");

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_scatter)
    .additional_info("eevee_shared")
    .uniform_buf(0, "DepthOfFieldData", "dof_buf")
    .sampler(0, ImageType::FLOAT_2D, "color_tx")
    .sampler(1, ImageType::DEPTH_2D, "depth_tx")
    .sampler(2, ImageType::FLOAT_2D, "occlusion_tx")
    .sampler(3, ImageType::FLOAT_2D, "coc_tx")
    .fragment_out(0, Type::VEC4, "fragColor")
    .vertex_out(eevee_depth_of_field_scatter_iface)
    .vertex_source("eevee_depth_of_field_scatter_vert.glsl")
    .fragment_source("eevee_depth_of_field_scatter_frag.glsl");

EEVEE_DOF_GROUND_VARIATIONS(eevee_depth_of_field_scatter, "eevee_depth_of_field_scatter")

/** \} */

/* -------------------------------------------------------------------- */
/** \name Resolve
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_resolve)
    .define("DOF_RESOLVE_PASS", "true")
    .additional_info("eevee_shared")
    .uniform_buf(0, "DepthOfFieldData", "dof_buf")
    .uniform_buf(1, "SamplingData", "sampling_buf")
    .sampler(0, ImageType::DEPTH_2D, "depth_tx")
    .sampler(1, ImageType::FLOAT_2D, "color_tx")
    .sampler(2, ImageType::FLOAT_2D, "color_bg_tx")
    .sampler(3, ImageType::FLOAT_2D, "color_fg_tx")
    .sampler(4, ImageType::FLOAT_2D, "color_holefill_tx")
    .sampler(7, ImageType::FLOAT_2D, "tiles_bg_tx")
    .sampler(8, ImageType::FLOAT_2D, "tiles_fg_tx")
    .sampler(9, ImageType::FLOAT_2D, "weight_bg_tx")
    .sampler(10, ImageType::FLOAT_2D, "weight_fg_tx")
    .sampler(11, ImageType::FLOAT_2D, "weight_holefill_tx")
    .fragment_out(0, Type::VEC4, "out_color")
    .fragment_source("eevee_depth_of_field_resolve_frag.glsl")
    .additional_info("draw_fullscreen");

EEVEE_DOF_HQ_VARIATIONS(eevee_depth_of_field_resolve, "eevee_depth_of_field_resolve")

/** \} */

/* -------------------------------------------------------------------- */
/** \name Circle-Of-Confusion Tiles
 * \{ */

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_tiles_dilate)
    .additional_info("eevee_shared")
    .sampler(0, ImageType::FLOAT_2D, "tiles_fg_tx", Frequency::PASS)
    .sampler(1, ImageType::FLOAT_2D, "tiles_bg_tx", Frequency::PASS)
    .fragment_out(0, Type::VEC4, "out_tile_fg")
    .fragment_out(1, Type::VEC3, "out_tile_bg")
    .push_constant(Type::INT, "ring_count")
    .push_constant(Type::INT, "ring_width_multiplier")
    .push_constant(Type::BOOL, "dilate_slight_focus")
    .fragment_source("eevee_depth_of_field_tiles_dilate_frag.glsl")
    .additional_info("draw_fullscreen");

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_tiles_dilate_minabs)
    .do_static_compilation(true)
    .define("DILATE_MODE_MIN_MAX", "false")
    .additional_info("eevee_depth_of_field_tiles_dilate");

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_tiles_dilate_minmax)
    .do_static_compilation(true)
    .define("DILATE_MODE_MIN_MAX", "true")
    .additional_info("eevee_depth_of_field_tiles_dilate");

GPU_SHADER_CREATE_INFO(eevee_depth_of_field_tiles_flatten)
    .do_static_compilation(true)
    .additional_info("eevee_shared")
    .sampler(0, ImageType::FLOAT_2D, "coc_tx", Frequency::PASS)
    .fragment_out(0, Type::VEC4, "out_tile_fg")
    .fragment_out(1, Type::VEC3, "out_tile_bg")
    .fragment_source("eevee_depth_of_field_tiles_flatten_frag.glsl")
    .additional_info("draw_fullscreen");

/** \} */
