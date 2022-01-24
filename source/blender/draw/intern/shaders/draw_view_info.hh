
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Draw View
 * \{ */

GPU_SHADER_CREATE_INFO(draw_view)
    .uniform_buf(0, "ViewInfos", "drw_view", Frequency::PASS)
    .typedef_source("draw_shader_shared.h");

GPU_SHADER_CREATE_INFO(draw_modelmat)
    .uniform_buf(8, "ObjectMatrices", "drw_matrices[DRW_RESOURCE_CHUNK_LEN]", Frequency::BATCH)
    .additional_info("draw_view");

GPU_SHADER_CREATE_INFO(draw_modelmat_legacy)
    .define("DRW_LEGACY_MODEL_MATRIX")
    .push_constant(38, Type::MAT4, "ModelMatrix")
    .push_constant(54, Type::MAT4, "ModelMatrixInverse")
    .additional_info("draw_view");

GPU_SHADER_CREATE_INFO(draw_modelmat_instanced_attr)
    .push_constant(0, Type::MAT4, "ModelMatrix")
    .push_constant(16, Type::MAT4, "ModelMatrixInverse")
    .additional_info("draw_view");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Draw View
 * \{ */

GPU_SHADER_CREATE_INFO(drw_clipped).define("USE_WORLD_CLIP_PLANES");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Geometry Type
 * \{ */

GPU_SHADER_CREATE_INFO(draw_mesh).additional_info("draw_modelmat");

GPU_SHADER_CREATE_INFO(draw_hair)
    .sampler(15, ImageType::FLOAT_BUFFER, "hairPointBuffer")
    .sampler(14, ImageType::UINT_BUFFER, "hairStrandBuffer")
    .sampler(13, ImageType::UINT_BUFFER, "hairStrandSegBuffer")
    /* TODO(fclem) Pack thoses into one UBO. */
    .push_constant(9, Type::INT, "hairStrandsRes")
    .push_constant(10, Type::INT, "hairThicknessRes")
    .push_constant(11, Type::FLOAT, "hairRadRoot")
    .push_constant(12, Type::FLOAT, "hairRadTip")
    .push_constant(13, Type::FLOAT, "hairRadShape")
    .push_constant(14, Type::BOOL, "hairCloseTip")
    .push_constant(15, Type::INT, "hairStrandOffset")
    .push_constant(16, Type::VEC4, "hairDupliMatrix", 4)
    .additional_info("draw_modelmat");

GPU_SHADER_CREATE_INFO(draw_pointcloud)
    .vertex_in(0, Type::VEC4, "pos")
    .vertex_in(1, Type::VEC3, "pos_inst")
    .vertex_in(2, Type::VEC3, "nor")
    .define("UNIFORM_RESOURCE_ID")
    .define("INSTANCED_ATTR")
    .additional_info("draw_modelmat_instanced_attr");

GPU_SHADER_CREATE_INFO(draw_volume).additional_info("draw_modelmat");

/** \} */
