
#pragma BLENDER_REQUIRE(common_view_lib.glsl)

/* UniformAttributes is defined by codegen. */

layout(std140) uniform uniformAttrs
{
  /* DRW_RESOURCE_CHUNK_LEN = 512 */
  UniformAttributes uniform_attrs[512];
};

#define UNIFORM_ATTR_UBO uniform_attrs[resource_id]
