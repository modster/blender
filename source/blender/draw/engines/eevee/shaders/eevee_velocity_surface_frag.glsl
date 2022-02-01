
/**
 * Output two 2D screen space velocity vector from object motion.
 * There is a separate output for view and camera vectors.
 * Camera vectors are used for reprojections and view vectors are used for motion blur fx.
 * xy: Previous position > Current position
 * zw: Current position  > Next position
 **/

#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_velocity_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

void main(void)
{
  compute_velocity(interp.P_prev,
                   interp.P,
                   interp.P_next,
                   cam_prev,
                   cam_curr,
                   cam_next,
                   out_velocity_camera,
                   out_velocity_view);
}
