#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_simple_lighting)
    .vertex_source("gpu_shader_3D_normal_vert.glsl")
    .fragment_source("gpu_shader_simple_lighting_frag.glsl")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(gpu_shader_2D_image_multi_rect_color)
    .vertex_source("datatoc_gpu_shader_2D_image_multi_rect_vert.glsl")
    .fragment_source("datatoc_gpu_shader_image_varying_color_frag.glsl")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(gpu_shader_2D_widget_base)
    .vertex_source("gpu_shader_2D_widget_base_vert.glsl")
    .fragment_source("gpu_shader_2D_widget_base_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_WIDGET_BASE] =
        {
            .name = "GPU_SHADER_2D_WIDGET_BASE",
            .vert = datatoc_gpu_shader_2D_widget_base_vert_glsl,
            .frag = datatoc_gpu_shader_2D_widget_base_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_widget_base_inst)
    .vertex_source("gpu_shader_2D_widget_base_vert.glsl")
    .fragment_source("gpu_shader_2D_widget_base_frag.glsl")
    .define("USE_INSTANCE")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_WIDGET_BASE_INST] =
        {
            .name = "GPU_SHADER_2D_WIDGET_BASE_INST",
            .vert = datatoc_gpu_shader_2D_widget_base_vert_glsl,
            .frag = datatoc_gpu_shader_2D_widget_base_frag_glsl,
            .defs = "#define USE_INSTANCE\n",
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_widget_shadow)
    .vertex_source("gpu_shader_2D_widget_shadow_vert.glsl")
    .fragment_source("gpu_shader_2D_widget_shadow_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_WIDGET_SHADOW] =
        {
            .name = "GPU_SHADER_2D_WIDGET_SHADOW",
            .vert = datatoc_gpu_shader_2D_widget_shadow_vert_glsl,
            .frag = datatoc_gpu_shader_2D_widget_shadow_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_gpencil_stroke)
    .vertex_source("gpu_shader_gpencil_stroke_vert.glsl")
    .geometry_source("gpu_shader_gpencil_stroke_geom.glsl")
    .fragment_source("gpu_shader_gpencil_stroke_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_GPENCIL_STROKE] = {
        .name = "GPU_SHADER_GPENCIL_STROKE",
        .vert = datatoc_gpu_shader_gpencil_stroke_vert_glsl,
        .geom = datatoc_gpu_shader_gpencil_stroke_geom_glsl,
        .frag = datatoc_gpu_shader_gpencil_stroke_frag_glsl,
},
*/