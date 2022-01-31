
/**
 * Special background shader for lookdev mode.
 * It samples the world cubemap at specified blur level and can fade to fully transparent.
 *
 * Note: This might becode obsolete if using glossy bsdf becomes supported in world shaders.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_cubemap_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

layout(std140) uniform lightprobes_info_block
{
  LightProbeInfoData probes_info;
};

uniform samplerCubeArray lightprobe_cube_tx;
uniform float opacity;
uniform float blur;

in vec4 uvcoordsvar;

layout(location = 0) out vec4 out_background;

void main()
{
  vec3 vP = get_view_space_from_depth(uvcoordsvar.xy, 0.5);
  vec3 vV = viewCameraVec(vP);
  vec3 R = -normal_view_to_world(vV);

  R = transform_direction(probes_info.cubes.lookdev_rotation, R);

  float lod = blur * probes_info.cubes.roughness_max_lod;
  out_background.rgb = cubemap_array_sample(lightprobe_cube_tx, vec4(R, 0), lod).rgb;
  out_background.rgb *= opacity;
  out_background.a = 1.0 - opacity;
}
