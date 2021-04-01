
/**
 * Shared structures, enum & defines between C++ and GLSL.
 */

/**
 * NOTE: Due to alignment restriction and buggy drivers, do not try to use vec3 or mat3 inside
 * structs. Use vec4 and pack an extra float at the end.
 * IMPORTANT: Don't forget to align mat4 and vec4 to 16 bytes.
 **/
#ifndef __cplusplus /* GLSL */
#  define STRUCT_TYPE_START(type_) struct type_
#  define STRUCT_TYPE_END(type_) ;
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

#else /* C++ */
#  pragma once
#  define STRUCT_TYPE_START(type_) typedef struct type_
#  define STRUCT_TYPE_END(type_) \
    type_; \
    BLI_STATIC_ASSERT_ALIGN(type_, 16);
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

STRUCT_TYPE_START(CameraData)
{
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
  float near_clip;
  float far_clip;
  /** Film pixel filter radius. */
  float filter_size;
  ENUM(eCameraType, type);
}
STRUCT_TYPE_END(CameraData)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Film
 * \{ */

ENUM_TYPE_START(eFilmDataType)
/** Color is accumulated using the pixel filter and pre-exposed. No negative values. */
ENUM_VAL(FILM_DATA_COLOR, 0)
/** Non-Color will be accumulated using nearest filter. All values are allowed. */
ENUM_VAL(FILM_DATA_FLOAT, 1)
ENUM_VAL(FILM_DATA_VEC2, 2)
/** No VEC3 because GPU_RGB16F is not a renderable format. */
ENUM_VAL(FILM_DATA_VEC4, 3)
ENUM_VAL(FILM_DATA_NORMAL, 4)
ENUM_VAL(FILM_DATA_DEPTH, 5)
ENUM_TYPE_END

STRUCT_TYPE_START(FilmData)
{
  /** Size of the render target. */
  IVEC2(extent);
  /** Data type stored by this film. */
  ENUM(eFilmDataType, data_type);
  /** Is true if history is valid and can be sampled. Bypassing history to resets accumulation. */
  BOOL(use_history);
}
STRUCT_TYPE_END(FilmData)

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