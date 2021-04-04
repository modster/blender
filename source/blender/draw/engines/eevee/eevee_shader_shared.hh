
/**
 * Shared structures, enums & defines between C++ and GLSL.
 */

/**
 * NOTE: Enum support is not part of GLSL. It is handled by our own pre-processor pass in
 * EEVEE's shader module.
 *
 * IMPORTANT: Don't add trailing comma at the end of the enum. Always use `u` suffix for values.
 * Define all values. Always use uint32_t as underlying type.
 *
 * NOTE: Due to alignment restriction and buggy drivers, do not try to use vec3 or mat3 inside
 * structs. Use vec4 and pack an extra float at the end.
 *
 * IMPORTANT: Don't forget to align mat4 and vec4 to 16 bytes.
 **/
#ifndef __cplusplus /* GLSL */
#  define BLI_STATIC_ASSERT_ALIGN(type_, align_)

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

#ifdef __cplusplus
#  undef bool
}  // namespace blender::eevee
#endif
