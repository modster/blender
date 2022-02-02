
/**
 * Extract two 2D screen space velocity vector from depth buffer.
 * Note that the offsets are in camera uv space, not view uv space.
 * xy: Previous position > Current position
 * zw: Current position  > Next position
 **/

#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_velocity_lib.glsl)

void main(void)
{
  float depth = textureLod(depth_tx, uvcoordsvar.xy, 0.0).r;

  vec3 P = get_world_space_from_depth(uvcoordsvar.xy, depth);
  vec3 P_prev = P, P_next = P;

  if (depth == 1.0) {
    /* Background case. Only compute rotation velocity. */
    vec3 V = -cameraVec(P);
    P_prev = cam_prev.viewinv[3].xyz + V;
    P_next = cam_next.viewinv[3].xyz + V;
  }

  if (depth == 1.0) {
    /* Lookdev Sphere. Ouput no velocity. */
    out_velocity_camera = vec4(0);
    out_velocity_view = vec4(0);
    return;
  }

  compute_velocity(
      P_prev, P, P_next, cam_prev, cam_curr, cam_next, out_velocity_camera, out_velocity_view);
}
