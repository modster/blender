
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(gpu_shader_codegen_lib.glsl)

struct GlobalData {
  /** UV to sample render layer. Cover the whole viewport. */
  vec2 uv_render_layer;
  /** UV of the compositing area. Also known as Camera Texture coordinates in material shaders. */
  vec2 uv_texco;
};

GlobalData g_data;

void ntree_eval_init()
{
}

void ntree_eval_weights()
{
}
