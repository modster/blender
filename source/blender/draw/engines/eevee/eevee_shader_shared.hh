
/**
 * Shared structures, enum & defines between C++ and GLSL.
 */

/**
 * NOTE: Due to alignment restriction and buggy drivers, do not try to use vec3 or mat3 inside
 * structs. Use vec4 and pack an extra float at the end.
 * IMPORTANT: Don't forget to align mat4 and vec4 to 16 bytes.
 **/
#ifndef __cplusplus /* GLSL */
#  define MAT4(member_) mat4 member_
#  define VEC4(member_) vec4 member_
#  define VEC2(member_) vec2 member_
#  define IVEC4(member_) ivec4 member_
#  define IVEC2(member_) ivec2 member_
#  define BOOL(member_) bool member_
#  define ENUM(type_, member_) int member_
#  define ENUM_TYPE_START(type_)
#  define ENUM_VAL(name_, value_) const int name_ = value_;
#  define ENUM_TYPE_END
#  define BLI_STATIC_ASSERT_ALIGN(type_, align_)

#else /* C++ */
#  pragma once
#  define MAT4(member_) float member_[4][4]
#  define VEC4(member_) float member_[4]
#  define VEC2(member_) float member_[2]
#  define IVEC4(member_) int member_[4]
#  define IVEC2(member_) int member_[2]
#  define BOOL(member_) int member_
#  define ENUM(type_, member_) type_ member_
#  define ENUM_TYPE_START(type_) enum type_ : int32_t {
#  define ENUM_VAL(name_, value_) name_ = value_,
/* Formatting is buggy here. */
/* clang-format off */
#  define ENUM_TYPE_END };
/* clang-format on */

namespace blender::eevee {

#endif

/* -------------------------------------------------------------------- */
/** \name Camera
 * \{ */

ENUM_TYPE_START(eCameraType)
ENUM_VAL(CAMERA_PERSP, 0)
ENUM_VAL(CAMERA_ORTHO, 1)
ENUM_VAL(CAMERA_PANO_EQUIRECT, 2)
ENUM_VAL(CAMERA_PANO_EQUISOLID, 3)
ENUM_VAL(CAMERA_PANO_EQUIDISTANT, 4)
ENUM_VAL(CAMERA_PANO_MIRROR, 5)
ENUM_TYPE_END

struct CameraData {
  /* View Matrices of the camera, not from any view! */
  MAT4(persmat);
  MAT4(persinv);
  MAT4(viewmat);
  MAT4(viewinv);
  MAT4(winmat);
  MAT4(wininv);
  /** Camera UV scale and bias. Also known as viewcamtexcofac. */
  VEC2(uv_scale);
  VEC2(uv_bias);
  /** Panorama parameters. */
  VEC2(equirect_scale);
  VEC2(equirect_scale_inv);
  VEC2(equirect_bias);
  float fisheye_fov;
  float fisheye_lens;
  /** Clipping distances. */
  float clip_near;
  float clip_far;
  /** Film pixel filter radius. */
  float filter_size;
  ENUM(eCameraType, type);
};
BLI_STATIC_ASSERT_ALIGN(CameraData, 16)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Film
 * \{ */

ENUM_TYPE_START(eFilmDataType)
/** Color is accumulated using the pixel filter. No negative values. */
ENUM_VAL(FILM_DATA_COLOR, 0)
/** Variant where we accumulate using pre-exposed values and log space. */
ENUM_VAL(FILM_DATA_COLOR_LOG, 1)
/** Non-Color will be accumulated using nearest filter. All values are allowed. */
ENUM_VAL(FILM_DATA_FLOAT, 2)
ENUM_VAL(FILM_DATA_VEC2, 3)
/** No VEC3 because GPU_RGB16F is not a renderable format. */
ENUM_VAL(FILM_DATA_VEC4, 4)
ENUM_VAL(FILM_DATA_NORMAL, 5)
ENUM_VAL(FILM_DATA_DEPTH, 6)
ENUM_TYPE_END

struct FilmData {
  /** Size of the render target in pixels. */
  IVEC2(extent);
  /** Offset of the render target in the full-res frame, in pixels. */
  IVEC2(offset);
  /** Scale and bias to filter only a region of the render (aka. render_border). */
  VEC2(uv_bias);
  VEC2(uv_scale);
  VEC2(uv_scale_inv);
  /** Data type stored by this film. */
  ENUM(eFilmDataType, data_type);
  /** Is true if history is valid and can be sampled. Bypassing history to resets accumulation. */
  BOOL(use_history);
};
BLI_STATIC_ASSERT_ALIGN(FilmData, 16)

/** \} */

#ifdef __cplusplus
}  // namespace blender::eevee
#endif

#undef STRUCT_TYPE_START
#undef STRUCT_TYPE_END
#undef MAT4
#undef VEC4
#undef VEC2
#undef IVEC4
#undef IVEC2
#undef BOOL
#undef ENUM
#undef ENUM_TYPE_START
#undef ENUM_VAL
#undef ENUM_TYPE_END