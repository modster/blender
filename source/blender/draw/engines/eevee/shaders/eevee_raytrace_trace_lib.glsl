/**
 * Screen-space raytracing routine.
 *
 * Based on "Efficient GPU Screen-Space Ray Tracing"
 * by Morgan McGuire & Michael Mara
 * https://jcgt.org/published/0003/04/04/paper.pdf
 *
 * Many modifications were made for our own usage.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

/**
 * As input to the tracing function, direction is premultiplied by its maximum length.
 * As output, direction is scaled to hit point or to latest step.
 */
struct Ray {
  vec3 origin;
  vec3 direction;
};

/**
 * Screenspace ray ([0..1] "uv" range) where direction is normalize to be as small as one
 * full-resolution pixel. The ray is also clipped to all frustum sides.
 */
struct ScreenSpaceRay {
  vec4 origin;
  vec4 direction;
  float max_time;
};

/* Inputs expected to be in viewspace. */
void raytrace_clip_ray_to_near_plane(inout Ray ray)
{
  float near_dist = get_view_z_from_depth(0.0);
  if ((ray.origin.z + ray.direction.z) > near_dist) {
    ray.direction *= abs((near_dist - ray.origin.z) / ray.direction.z);
  }
}

void raytrace_screenspace_ray_finalize(HiZData hiz, inout ScreenSpaceRay ray)
{
  /* Constant bias (due to depth buffer precision). Helps with self intersection. */
  /* Magic numbers for 24bits of precision.
   * From http://terathon.com/gdc07_lengyel.pdf (slide 26) */
  const float bias = -2.4e-7 * 2.0;
  ray.origin.zw += bias;
  ray.direction.zw += bias;

  ray.direction -= ray.origin;
  /* If the line is degenerate, make it cover at least one pixel
   * to not have to handle zero-pixel extent as a special case later. */
  if (len_squared(ray.direction.xy) < 0.00001) {
    ray.direction.xy = vec2(0.0, 0.00001);
  }
  float ray_len_sqr = len_squared(ray.direction.xyz);
  /* Make ray.direction cover one pixel. */
  bool is_more_vertical = abs(ray.direction.x) < abs(ray.direction.y);
  ray.direction /= (is_more_vertical) ? abs(ray.direction.y) : abs(ray.direction.x);
  ray.direction *= (is_more_vertical) ? hiz.pixel_to_ndc.y : hiz.pixel_to_ndc.x;
  /* Clip to segment's end. */
  ray.max_time = sqrt(ray_len_sqr * safe_rcp(len_squared(ray.direction.xyz)));
  /* Clipping to frustum sides. */
  float clip_dist = line_unit_box_intersect_dist_safe(ray.origin.xyz, ray.direction.xyz);
  ray.max_time = min(ray.max_time, clip_dist);
  /* Convert to texture coords [0..1] range. */
  ray.origin = ray.origin * 0.5 + 0.5;
  ray.direction *= 0.5;
}

ScreenSpaceRay raytrace_screenspace_ray_create(HiZData hiz, Ray ray)
{
  ScreenSpaceRay ssray;
  ssray.origin.xyz = project_point(ProjectionMatrix, ray.origin);
  ssray.direction.xyz = project_point(ProjectionMatrix, ray.origin + ray.direction);

  raytrace_screenspace_ray_finalize(hiz, ssray);
  return ssray;
}

ScreenSpaceRay raytrace_screenspace_ray_create(HiZData hiz, Ray ray, float thickness)
{
  ScreenSpaceRay ssray;
  ssray.origin.xyz = project_point(ProjectionMatrix, ray.origin);
  ssray.direction.xyz = project_point(ProjectionMatrix, ray.origin + ray.direction);
  /* Interpolate thickness in screen space.
   * Calculate thickness further away to avoid near plane clipping issues. */
  ssray.origin.w = get_depth_from_view_z(ray.origin.z - thickness);
  ssray.direction.w = get_depth_from_view_z(ray.origin.z + ray.direction.z - thickness);
  ssray.origin.w = ssray.origin.w * 2.0 - 1.0;
  ssray.direction.w = ssray.direction.w * 2.0 - 1.0;

  raytrace_screenspace_ray_finalize(hiz, ssray);
  return ssray;
}

/**
 * Raytrace against the given hizbuffer heightfield.
 *
 * \param stride_rand: Random number in [0..1] range. Offset along the ray to avoid banding
 *                     artifact when steps are too large.
 * \param roughness: Determine how lower depth mipmaps are used to make the tracing faster. Lower
 *                   roughness will use lower mipmaps.
 * \param discard_backface: If true, raytrace will return false  if we hit a surface from behind.
 * \param allow_self_intersection: If false, raytrace will return false if the ray is not covering
 *                                 at least one pixel.
 * \param ray: Viewspace ray. Direction premultiplied by maximum length.
 *
 * \return True if there is a valid intersection.
 */
bool raytrace_screen(RaytraceData raytrace,
                     HiZData hiz,
                     sampler2D hiz_tx,
                     float stride_rand,
                     float roughness,
                     const bool discard_backface,
                     const bool allow_self_intersection,
                     inout Ray ray)
{
  /* Clip to near plane for perspective view where there is a singularity at the camera origin. */
  if (ProjectionMatrix[3][3] == 0.0) {
    raytrace_clip_ray_to_near_plane(ray);
  }

  ScreenSpaceRay ssray = raytrace_screenspace_ray_create(hiz, ray, raytrace.thickness);
  /* Avoid no iteration. */
  if (!allow_self_intersection && ssray.max_time < 1.1) {
    return false;
  }

  ssray.max_time = max(1.1, ssray.max_time);

  float prev_delta = 0.0, prev_time = 0.0;
  float depth_sample = get_depth_from_view_z(ray.origin.z);
  float delta = depth_sample - ssray.origin.z;

  float lod_fac = saturate(fast_sqrt(roughness) * 2.0 - 0.4);

  /* Cross at least one pixel. */
  float t = 1.001, time = 1.001;
  bool hit = false;
  const float max_steps = 255.0;
  for (float iter = 1.0; !hit && (time < ssray.max_time) && (iter < max_steps); iter++) {
    float stride = 1.0 + iter * raytrace.quality;
    float lod = log2(stride) * lod_fac;

    prev_time = time;
    prev_delta = delta;

    time = min(t + stride * stride_rand, ssray.max_time);
    t += stride;

    vec4 ss_p = ssray.origin + ssray.direction * time;
    depth_sample = textureLod(hiz_tx, ss_p.xy * hiz.uv_scale, floor(lod)).r;

    delta = depth_sample - ss_p.z;
    /* Check if the ray is below the surface ... */
    hit = (delta < 0.0);
    /* ... and above it with the added thickness. */
    hit = hit && (delta > ss_p.z - ss_p.w || abs(delta) < abs(ssray.direction.z * stride * 2.0));
  }
  /* Discard backface hits. */
  hit = hit && !(discard_backface && prev_delta < 0.0);
  /* Reject hit if background. */
  hit = hit && (depth_sample != 1.0);
  /* Refine hit using intersection between the sampled heightfield and the ray.
   * This simplifies nicely to this single line. */
  time = mix(prev_time, time, saturate(prev_delta / (prev_delta - delta)));

  vec3 hit_ssP = ssray.origin.xyz + ssray.direction.xyz * time;
  /* Set ray to where tracing ended. */
  vec3 hit_P = get_view_space_from_depth(hit_ssP.xy, saturate(hit_ssP.z));
  ray.direction = hit_P - ray.origin;

  return hit;
}