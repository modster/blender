
#include "gpu_shader_create_info.hh"

/* -------------------------------------------------------------------- */
/** \name Debug print
 *
 * Allows print() function to have logging support inside shaders.
 * \{ */

GPU_SHADER_CREATE_INFO(draw_debug_print)
    .define("DRAW_DEBUG_PRINT_DATA")
    .storage_buf(15, Qualifier::READ_WRITE, "uint", "drw_print_buf[]", Frequency::PASS);

GPU_SHADER_INTERFACE_INFO(draw_debug_print_display_iface, "").flat(Type::UINT, "char_index");

GPU_SHADER_CREATE_INFO(draw_debug_print_display)
    .do_static_compilation(true)
    .vertex_in(0, Type::UINT, "char_data")
    .vertex_out(draw_debug_print_display_iface)
    .fragment_out(0, Type::VEC4, "out_color")
    .vertex_source("draw_debug_print_display_vert.glsl")
    .fragment_source("draw_debug_print_display_frag.glsl")
    .additional_info("draw_view");

/** \} */

/* -------------------------------------------------------------------- */
/** \name Debug draw shapes
 *
 * Allows to draw lines and points just like the DRW_debug module functions.
 * \{ */

GPU_SHADER_CREATE_INFO(draw_debug_draw)
    .define("DRW_DEBUG_DRAW")
    .typedef_source("draw_shader_shared.h")
    .storage_buf(15, Qualifier::READ_WRITE, "DebugVert", "drw_debug_verts[]", Frequency::PASS);

/** \} */
