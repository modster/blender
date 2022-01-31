
/**
 * Apply transmittance to all radiance passes.
 *
 * We needs to evaluate the transmittance of homogeneous volumes if any is present.
 * Hopefully, this has O(1) complexity as we do not need to raymarch the volume.
 *
 * Using blend mode multiply.
 **/

#pragma BLENDER_REQUIRE(eevee_volume_eval_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_gbuffer_lib.glsl)

uniform usampler2D volume_data_tx;
uniform sampler2D transparency_data_tx;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_combined;
layout(location = 1) out vec3 out_diffuse;
layout(location = 2) out vec3 out_specular;
layout(location = 3) out vec3 out_volume;
layout(location = 4) out vec3 out_background;
layout(location = 5) out vec3 out_holdout;

void main(void)
{
  ClosureVolume volume_data = gbuffer_load_volume_data(volume_data_tx, uvcoordsvar.xy);

  /* For volumes from solid objects. */
  // float depth_max = linear_z(texture(depth_max_tx, uv).r);
  // float depth_min = linear_z(texture(depth_min_tx, uv).r);

  /* Refine bounds to skip empty areas. */
  // float dist_from_bbox = intersect_bbox_ray(P, V, bbox);
  // depth_min = max(dist_from_bbox, depth_min);

  vec3 volume_transmittance;
  if (volume_data.anisotropy == VOLUME_HETEROGENEOUS) {
    volume_transmittance = volume_data.transmittance;
  }
  else {
    // volume_eval_homogeneous(P, depth_min, depth_max, volume_transmittance);
    volume_transmittance = vec3(0.0);
  }

  vec3 surface_transmittance =
      gbuffer_load_transparency_data(transparency_data_tx, uvcoordsvar.xy).transmittance;

  vec3 final_transmittance = volume_transmittance * surface_transmittance;

  /* Multiply transmittance all radiance buffers. Remember that blend mode is multiply. */
  out_combined = vec4(final_transmittance, avg(final_transmittance));
  out_diffuse = final_transmittance;
  out_specular = final_transmittance;
  out_volume = final_transmittance;
  out_background = final_transmittance;
  out_holdout = final_transmittance;
}
