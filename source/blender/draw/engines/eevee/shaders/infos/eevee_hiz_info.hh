
/* FIXME(@fclem): This file is included inside the gpu module. We have to workaround to include
 * eevee header. */
#include "../../draw/engines/eevee/eevee_defines.hh"

#include "gpu_shader_create_info.hh"

/** NOTE: Read depth format, output color format. */
GPU_SHADER_CREATE_INFO(eevee_hiz_update)
    .do_static_compilation(true)
    .local_group_size(HIZ_GROUP_SIZE, HIZ_GROUP_SIZE)
    .sampler(0, ImageType::DEPTH_2D, "depth_tx", Frequency::PASS, GPU_SAMPLER_FILTER)
    .image(0, GPU_R32F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_lvl0")
    .image(1, GPU_R32F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_lvl1")
    .image(2, GPU_R32F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_lvl2")
    .image(3, GPU_R32F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_lvl3")
    .image(4, GPU_R32F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_lvl4")
    .image(5, GPU_R32F, Qualifier::WRITE, ImageType::FLOAT_2D, "out_lvl5")
    .compute_source("eevee_hiz_update_comp.glsl");
