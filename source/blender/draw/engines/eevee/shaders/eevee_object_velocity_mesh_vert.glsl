
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_object_velocity_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform object_block
{
  VelocityObjectData velocity;
};

uniform int data_offset;

in vec3 pos;
in vec3 prv;
in vec3 nxt;

vec3 velocity_object_to_world_prev(VelocityObjectData data, vec3 prev_pos, vec3 current_pos)
{
  /* Encoded use_deform inside the matrix to save up space. */
  bool use_deform = data.next_object_mat[3][3] == 0.0;
  return transform_point(data.prev_object_mat, use_deform ? prev_pos : current_pos);
}

vec3 velocity_object_to_world_next(VelocityObjectData data, vec3 next_pos, vec3 current_pos)
{
  /* Encoded use_deform inside the matrix to save up space. */
  bool use_deform = data.next_object_mat[3][3] == 0.0;
  mat4 obmat = data.next_object_mat;
  obmat[3][3] = 1.0;
  return transform_point(obmat, use_deform ? next_pos : current_pos);
}

void main(void)
{
  interp.P = point_object_to_world(pos);
  interp.P_prev = velocity_object_to_world_prev(velocity, prv, pos);
  interp.P_next = velocity_object_to_world_next(velocity, nxt, pos);

  gl_Position = point_world_to_ndc(interp.P);
}