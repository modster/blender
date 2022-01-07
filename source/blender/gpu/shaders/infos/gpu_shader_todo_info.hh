#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_2D_point_varying_size_varying_color)
    .vertex_source("gpu_shader_2D_point_varying_size_varying_color_vert.glsl")
    .fragment_source("gpu_shader_point_varying_color_frag.glsl")
    .do_static_compilation(true);

GPU_SHADER_CREATE_INFO(gpu_shader_2d_point_uniform_size_uniform_color_aa)
    .vertex_source("gpu_shader_2D_point_uniform_size_aa_vert.glsl")
    .fragment_source("gpu_shader_point_uniform_color_aa_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_AA] =
        {
            .name = "GPU_SHADER_2D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_AA",
            .vert = datatoc_gpu_shader_2D_point_uniform_size_aa_vert_glsl,
            .frag = datatoc_gpu_shader_point_uniform_color_aa_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2d_point_uniform_size_uniform_color_outline_aa)
    .vertex_source("gpu_shader_2D_point_uniform_size_outline_aa_vert.glsl")
    .fragment_source("gpu_shader_point_uniform_color_outline_aa_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_OUTLINE_AA] =
        {
            .name = "GPU_SHADER_2D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_OUTLINE_AA",
            .vert = datatoc_gpu_shader_2D_point_uniform_size_outline_aa_vert_glsl,
            .frag = datatoc_gpu_shader_point_uniform_color_outline_aa_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2d_point_uniform_size_varying_color_outline_aa)
    .vertex_source("gpu_shader_2D_point_uniform_size_varying_color_outline_aa_vert.glsl")
    .fragment_source("gpu_shader_point_varying_color_outline_aa_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_POINT_UNIFORM_SIZE_VARYING_COLOR_OUTLINE_AA] =
        {
            .name = "GPU_SHADER_2D_POINT_UNIFORM_SIZE_VARYING_COLOR_OUTLINE_AA",
            .vert = datatoc_gpu_shader_2D_point_uniform_size_varying_color_outline_aa_vert_glsl,
            .frag = datatoc_gpu_shader_point_varying_color_outline_aa_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_3D_point_fixed_size_uniform_color)
    .vertex_source("gpu_shader_3D_vert.glsl")
    .fragment_source("gpu_shader_point_uniform_color_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_3D_POINT_FIXED_SIZE_UNIFORM_COLOR] =
        {
            .name = "GPU_SHADER_3D_POINT_FIXED_SIZE_UNIFORM_COLOR",
            .vert = datatoc_gpu_shader_3D_vert_glsl,
            .frag = datatoc_gpu_shader_point_uniform_color_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_3D_point_fixed_size_varying_color)
    .vertex_source("gpu_shader_3D_point_fixed_size_varying_color_vert.glsl")
    .fragment_source("gpu_shader_point_varying_color_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_3D_POINT_FIXED_SIZE_VARYING_COLOR] =
        {
            .name = "GPU_SHADER_3D_POINT_FIXED_SIZE_VARYING_COLOR",
            .vert = datatoc_gpu_shader_3D_point_fixed_size_varying_color_vert_glsl,
            .frag = datatoc_gpu_shader_point_varying_color_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_3D_point_varying_size_uniform_color)
    .vertex_source("gpu_shader_3D_point_varying_size_vert.glsl")
    .fragment_source("gpu_shader_point_uniform_color_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_3D_POINT_VARYING_SIZE_UNIFORM_COLOR] =
        {
            .name = "GPU_SHADER_3D_POINT_VARYING_SIZE_UNIFORM_COLOR",
            .vert = datatoc_gpu_shader_3D_point_varying_size_vert_glsl,
            .frag = datatoc_gpu_shader_point_uniform_color_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_3D_point_varying_size_varying_color)
    .vertex_source("gpu_shader_3D_point_varying_size-varying_color_vert.glsl")
    .fragment_source("gpu_shader_point_varying_color_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_3D_POINT_VARYING_SIZE_VARYING_COLOR] =
        {
            .name = "GPU_SHADER_3D_POINT_VARYING_SIZE_VARYING_COLOR",
            .vert = datatoc_gpu_shader_3D_point_varying_size_varying_color_vert_glsl,
            .frag = datatoc_gpu_shader_point_varying_color_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_3D_point_uniform_size_uniform_color_aa)
    .vertex_source("gpu_shader_3D_point_uniform_size_aa_vert.glsl")
    .fragment_source("gpu_shader_point_uniform_color_aa_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_3D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_AA] =
        {
            .name = "GPU_SHADER_3D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_AA",
            .vert = datatoc_gpu_shader_3D_point_uniform_size_aa_vert_glsl,
            .frag = datatoc_gpu_shader_point_uniform_color_aa_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_3D_point_uniform_size_uniform_color_outline_aa)
    .vertex_source("gpu_shader_3D_point_uniform_size_outline_aa_vert.glsl")
    .fragment_source("gpu_shader_point_uniform_color_outline_aa_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_3D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_OUTLINE_AA] =
        {
            .name = "GPU_SHADER_3D_POINT_UNIFORM_SIZE_UNIFORM_COLOR_OUTLINE_AA",
            .vert = datatoc_gpu_shader_3D_point_uniform_size_outline_aa_vert_glsl,
            .frag = datatoc_gpu_shader_point_uniform_color_outline_aa_frag_glsl,
},

*/
GPU_SHADER_CREATE_INFO(gpu_shader_instance_varying_color_varying_size)
    .vertex_source("gpu_shader_instance_varying_size_varying_color_vert.glsl")
    .fragment_source("gpu_shader_flat_color_frag.glsl")
    .define("UNIFORM_SCALED")
    .do_static_compilation(true);
/*
    [GPU_SHADER_INSTANCE_VARIYING_COLOR_VARIYING_SIZE] =
        {
            .name = "GPU_SHADER_INSTANCE_VARIYING_COLOR_VARIYING_SIZE",
            .vert = datatoc_gpu_shader_instance_variying_size_variying_color_vert_glsl,
            .frag = datatoc_gpu_shader_flat_color_frag_glsl,
            .defs = "#define UNIFORM_SCALE\n",
},

*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_area_edges)
    .vertex_source("gpu_shader_2D_area_borders_vert.glsl")
    .fragment_source("gpu_shader_2D_area_borders_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_AREA_EDGES] =
        {
            .name = "GPU_SHADER_2D_AREA_EDGES",
            .vert = datatoc_gpu_shader_2D_area_borders_vert_glsl,
            .frag = datatoc_gpu_shader_2D_area_borders_frag_glsl,
},
*/
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
GPU_SHADER_CREATE_INFO(gpu_shader_2D_nodelink)
    .vertex_source("gpu_shader_2D_nodelink_vert.glsl")
    .fragment_source("gpu_shader_2D_nodelink_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_NODELINK] =
        {
            .name = "GPU_SHADER_2D_NODELINK",
            .vert = datatoc_gpu_shader_2D_nodelink_vert_glsl,
            .frag = datatoc_gpu_shader_2D_nodelink_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_nodelink_inst)
    .vertex_source("gpu_shader_2D_nodelink_vert.glsl")
    .fragment_source("gpu_shader_2D_nodelink_frag.glsl")
    .define("USE_INSTANCE")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_NODELINK_INST] =
        {
            .name = "GPU_SHADER_2D_NODELINK_INST",
            .vert = datatoc_gpu_shader_2D_nodelink_vert_glsl,
            .frag = datatoc_gpu_shader_2D_nodelink_frag_glsl,
            .defs = "#define USE_INSTANCE\n",
},

*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_uniform_color)
    .vertex_source("gpu_shader_2D_vert.glsl")
    .fragment_source("gpu_shader_uniform_color_frag.glsl")
    .define("UV_POS")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_UV_UNIFORM_COLOR] =
        {
            .name = "GPU_SHADER_2D_UV_UNIFORM_COLOR",
            .vert = datatoc_gpu_shader_2D_vert_glsl,
            .frag = datatoc_gpu_shader_uniform_color_frag_glsl,
            .defs = "#define UV_POS\n",
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_verts)
    .vertex_source("gpu_shader_2D_edituvs_points_vert.glsl")
    .fragment_source("gpu_shader_point_varying_color_varying_outline_aa_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_UV_VERTS] =
        {
            .name = "GPU_SHADER_2D_UV_VERTS",
            .vert = datatoc_gpu_shader_2D_edituvs_points_vert_glsl,
            .frag = datatoc_gpu_shader_point_varying_color_varying_outline_aa_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_facedots)
    .vertex_source("gpu_shader_2D_edituvs_facedots_vert.glsl")
    .fragment_source("gpu_shader_point_varying_color_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_UV_FACEDOTS] =
        {
            .name = "GPU_SHADER_2D_UV_FACEDOTS",
            .vert = datatoc_gpu_shader_2D_edituvs_facedots_vert_glsl,
            .frag = datatoc_gpu_shader_point_varying_color_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_edges)
    .vertex_source("gpu_shader_2D_edituvs_edges_vert.glsl")
    .fragment_source("gpu_shader_2D_edituvs_edges_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_UV_EDGES] =
        {
            .name = "GPU_SHADER_2D_UV_EDGES",
            .vert = datatoc_gpu_shader_2D_edituvs_edges_vert_glsl,
            .frag = datatoc_gpu_shader_2D_edituvs_edges_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_edges_smooth)
    .vertex_source("gpu_shader_2D_edituvs_edges_vert.glsl")
    .fragment_source("gpu_shader_2D_edituvs_edges_frag.glsl")
    .define("SMOOTH_COLOR")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_UV_EDGES_SMOOTH] =
        {
            .name = "GPU_SHADER_2D_UV_EDGES_SMOOTH",
            .vert = datatoc_gpu_shader_2D_edituvs_edges_vert_glsl,
            .frag = datatoc_gpu_shader_2D_edituvs_edges_frag_glsl,
            .defs = "#define SMOOTH_COLOR\n",
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_faces)
    .vertex_source("gpu_shader_2D_edituvs_faces_vert.glsl")
    .fragment_source("gpu_shader_flat_color_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_UV_FACES] =
        {
            .name = "GPU_SHADER_2D_UV_FACES",
            .vert = datatoc_gpu_shader_2D_edituvs_faces_vert_glsl,
            .frag = datatoc_gpu_shader_flat_color_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_faces_stretch_area)
    .vertex_source("gpu_shader_2D_edituvs_stretch_vert.glsl")
    .fragment_source("gpu_shader_2D_smooth_color_frag.glsl")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_UV_FACES_STRETCH_AREA] =
        {
            .name = "GPU_SHADER_2D_UV_FACES_STRETCH_AREA",
            .vert = datatoc_gpu_shader_2D_edituvs_stretch_vert_glsl,
            .frag = datatoc_gpu_shader_2D_smooth_color_frag_glsl,
},
*/
GPU_SHADER_CREATE_INFO(gpu_shader_2D_uv_faces_stretch_angle)
    .vertex_source("gpu_shader_2D_edituvs_stretch_vert.glsl")
    .fragment_source("gpu_shader_2D_smooth_color_frag.glsl")
    .define("STRETCH_ANGLE")
    .do_static_compilation(true);
/*
    [GPU_SHADER_2D_UV_FACES_STRETCH_ANGLE] =
        {
            .name = "GPU_SHADER_2D_UV_FACES_STRETCH_ANGLE",
            .vert = datatoc_gpu_shader_2D_edituvs_stretch_vert_glsl,
            .frag = datatoc_gpu_shader_2D_smooth_color_frag_glsl,
            .defs = "#define STRETCH_ANGLE\n",
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