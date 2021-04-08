
/**
 * Shared structures, enums & defines between C++ and GLSL.
 * Can also include some math functions but they need to be simple enough to be valid in both
 * language.
 */

/**
 * NOTE: Enum support is not part of GLSL. It is handled by our own pre-processor pass in
 * EEVEE's shader module.
 *
 * IMPORTANT:
 * - Don't add trailing comma at the end of the enum. Our custom pre-processor will noy trim it
 *   for GLSL.
 * - Always use `u` suffix for values. GLSL do not support implicit cast.
 * - Define all values. This is in order to simplify custom pre-processor code.
 * - Always use uint32_t as underlying type.
 * - Use float suffix by default for float literals to avoid double promotion in C++.
 *
 * NOTE: Due to alignment restriction and buggy drivers, do not try to use vec3 or mat3 inside
 * structs. Use vec4 and pack an extra float at the end.
 * Do not use arrays of float. They are padded to arrays of vec4 and are not worth it.
 *
 * IMPORTANT: Don't forget to align mat4 and vec4 to 16 bytes.
 **/

#pragma BLI_STATIC_ASSERT_ALIGN(type_, align_)

#ifndef __cplusplus /* GLSL */
#  pragma BLENDER_REQUIRE(common_math_lib.glsl)
#  define BLI_STATIC_ASSERT_ALIGN(type_, align_)
#  define static
#  define cosf cos
#  define sinf sin
#  define tanf tan
#  define acosf acos
#  define asinf asin
#  define atanf atan
#  define floorf floor
#  define ceilf ceil
#  define sqrtf sqrt

#else /* C++ */
#  pragma once
/* TODO(fclem) Use correct C++ vector classes instead. */
typedef float mat4[4][4];
typedef float vec4[4];
typedef float vec2[2];
typedef int ivec4[4];
typedef int ivec2[2];
typedef int bvec4[4];
typedef int bvec2[2];
/* Ugly but it does the job! */
#  define bool int

namespace blender::eevee {

#endif

/* -------------------------------------------------------------------- */
/** \name Sampling
 * \{ */

enum eSamplingDimension : uint32_t {
  SAMPLING_FILTER_U = 0u,
  SAMPLING_FILTER_V = 1u,
  SAMPLING_LENS_U = 2u,
  SAMPLING_LENS_V = 3u,
  SAMPLING_TIME = 4u
};

struct SamplingData {
  /** Array containing random values from Low Discrepency Sequence in [0..1) range. */
  /** IMPORTANT: Make sure the array can contain all sampling dimensions. */
  /** HACK: float arrays are padded to vec4 in GLSL. Using vec4 for now to get the same alignment
   * but this is wasteful. */
  vec4 dimensions[8];
};
BLI_STATIC_ASSERT_ALIGN(SamplingData, 16)

/* Returns total sample count in a web pattern of the given size. */
static int web_sample_count_get(int web_density, int ring_count)
{
  return ((ring_count * ring_count + ring_count) / 2) * web_density + 1;
}

/* Returns lowest possible ring count that contains at least sample_count samples. */
static int web_ring_count_get(int web_density, int sample_count)
{
  /* Inversion of web_sample_count_get(). */
  float x = 2.0f * (float(sample_count) - 1.0f) / float(web_density);
  /* Solving polynomial. We only search positive solution. */
  float discriminant = 1.0f + 4.0f * x;
  return int(ceilf(0.5f * (sqrtf(discriminant) - 1.0f)));
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Camera
 * \{ */

enum eCameraType : uint32_t {
  CAMERA_PERSP = 0u,
  CAMERA_ORTHO = 1u,
  CAMERA_PANO_EQUIRECT = 2u,
  CAMERA_PANO_EQUISOLID = 3u,
  CAMERA_PANO_EQUIDISTANT = 4u,
  CAMERA_PANO_MIRROR = 5u
};

static bool is_panoramic(eCameraType type)
{
  return type > CAMERA_ORTHO;
}

struct CameraData {
  /* View Matrices of the camera, not from any view! */
  mat4 persmat;
  mat4 persinv;
  mat4 viewmat;
  mat4 viewinv;
  mat4 winmat;
  mat4 wininv;
  /** Camera UV scale and bias. Also known as viewcamtexcofac. */
  vec2 uv_scale;
  vec2 uv_bias;
  /** Panorama parameters. */
  vec2 equirect_scale;
  vec2 equirect_scale_inv;
  vec2 equirect_bias;
  float fisheye_fov;
  float fisheye_lens;
  /** Clipping distances. */
  float clip_near;
  float clip_far;
  /** Film pixel filter radius. */
  float filter_size;
  eCameraType type;
};
BLI_STATIC_ASSERT_ALIGN(CameraData, 16)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Film
 * \{ */

enum eFilmDataType : uint32_t {
  /** Color is accumulated using the pixel filter. No negative values. */
  FILM_DATA_COLOR = 0u,
  /** Variant where we accumulate using pre-exposed values and log space. */
  FILM_DATA_COLOR_LOG = 1u,
  /** Non-Color will be accumulated using nearest filter. All values are allowed. */
  FILM_DATA_FLOAT = 2u,
  FILM_DATA_VEC2 = 3u,
  /** No VEC3 because GPU_RGB16F is not a renderable format. */
  FILM_DATA_VEC4 = 4u,
  FILM_DATA_NORMAL = 5u,
  FILM_DATA_DEPTH = 6u
};

struct FilmData {
  /** Size of the render target in pixels. */
  ivec2 extent;
  /** Offset of the render target in the full-res frame, in pixels. */
  ivec2 offset;
  /** Scale and bias to filter only a region of the render (aka. render_border). */
  vec2 uv_bias;
  vec2 uv_scale;
  vec2 uv_scale_inv;
  /** Data type stored by this film. */
  eFilmDataType data_type;
  /** Is true if history is valid and can be sampled. Bypassing history to resets accumulation. */
  bool use_history;
};
BLI_STATIC_ASSERT_ALIGN(FilmData, 16)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Depth of field
 * \{ */

/* 5% error threshold. */
#define DOF_FAST_GATHER_COC_ERROR 0.05
#define DOF_GATHER_RING_COUNT 5
#define DOF_DILATE_RING_COUNT 3
#define DOF_TILE_DIVISOR 16
#define DOF_BOKEH_LUT_SIZE 32

struct DepthOfFieldData {
  /** Size of the render targets for gather & scatter passes. */
  ivec2 extent;
  /** Size of a pixel in uv space (1.0 / extent). */
  vec2 texel_size;
  /** Bokeh Scale factor. */
  vec2 bokeh_anisotropic_scale;
  vec2 bokeh_anisotropic_scale_inv;
  /* Correction factor to align main target pixels with the filtered mipmap chain texture. */
  vec2 gather_uv_fac;
  /** Scatter parameters. */
  float scatter_coc_threshold;
  float scatter_color_threshold;
  float scatter_neighbor_max_color;
  int scatter_sprite_per_row;
  /** Downsampling paramters. */
  float denoise_factor;
  /** Bokeh Shape parameters. */
  float bokeh_blades;
  float bokeh_rotation;
  /** Circle of confusion (CoC) parameters. */
  float coc_mul;
  float coc_bias;
  float coc_abs_max;
  /** Copy of camera type. */
  eCameraType camera_type;
  int pad0, pad1, pad2;
};
BLI_STATIC_ASSERT_ALIGN(DepthOfFieldData, 16)

static float coc_radius_from_camera_depth(DepthOfFieldData dof, float depth)
{
  depth = (dof.camera_type != CAMERA_ORTHO) ? 1.0f / depth : depth;
  return dof.coc_mul * depth + dof.coc_bias;
}

static float regular_polygon_side_length(float sides_count)
{
  return 2.0f * sinf(M_PI / sides_count);
}

/* Returns intersection ratio between the radius edge at theta and the regular polygon edge.
 * Start first corners at theta == 0. */
static float circle_to_polygon_radius(float sides_count, float theta)
{
  /* From Graphics Gems from CryENGINE 3 (Siggraph 2013) by Tiago Sousa (slide
   * 36). */
  float side_angle = (2.0f * M_PI) / sides_count;
  return cosf(side_angle * 0.5f) /
         cosf(theta - side_angle * floorf((sides_count * theta + M_PI) / (2.0f * M_PI)));
}

/* Remap input angle to have homogenous spacing of points along a polygon edge.
 * Expects theta to be in [0..2pi] range. */
static float circle_to_polygon_angle(float sides_count, float theta)
{
  float side_angle = (2.0f * M_PI) / sides_count;
  float halfside_angle = side_angle * 0.5f;
  float side = floorf(theta / side_angle);
  /* Length of segment from center to the middle of polygon side. */
  float adjacent = circle_to_polygon_radius(sides_count, 0.0f);

  /* This is the relative position of the sample on the polygon half side. */
  float local_theta = theta - side * side_angle;
  float ratio = (local_theta - halfside_angle) / halfside_angle;

  float halfside_len = regular_polygon_side_length(sides_count) * 0.5f;
  float opposite = ratio * halfside_len;

  /* NOTE: atan(y_over_x) has output range [-M_PI_2..M_PI_2]. */
  float final_local_theta = atanf(opposite / adjacent);

  return side * side_angle + final_local_theta;
}

/** \} */

#ifdef __cplusplus
#  undef bool
}  // namespace blender::eevee
#endif
