
#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)
#pragma BLENDER_REQUIRE(eevee_camera_lib.glsl)

void compute_velocity(vec3 P_prev,
                      vec3 P,
                      vec3 P_next,
                      CameraData cam_prev,
                      CameraData cam_curr,
                      CameraData cam_next,
                      out vec4 velocity_camera,
                      out vec4 velocity_view)
{
  vec2 prev_uv, curr_uv, next_uv;
  prev_uv = camera_uv_from_world(cam_prev, P_prev);
  curr_uv = camera_uv_from_world(cam_curr, P);
  next_uv = camera_uv_from_world(cam_next, P_next);

  velocity_camera.xy = prev_uv - curr_uv;
  velocity_camera.zw = curr_uv - next_uv;

  if (is_panoramic(cam_curr.type)) {
    /* This path is only used if using using panoramic projections. Since the views always have
     * the same 45° aperture angle, we can safely reuse the projection matrix. */
    prev_uv = transform_point(ProjectionMatrix, transform_point(cam_prev.viewmat, P_prev)).xy;
    curr_uv = transform_point(ViewProjectionMatrix, P).xy;
    next_uv = transform_point(ProjectionMatrix, transform_point(cam_next.viewmat, P_next)).xy;

    velocity_view.xy = prev_uv - curr_uv;
    velocity_view.zw = curr_uv - next_uv;
    /* Convert NDC velocity to UV velocity */
    velocity_view *= 0.5;
  }
  else {
    velocity_view = velocity_camera;
  }
}
