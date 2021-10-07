
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
 * - Pack one float or int after a vec3/ivec3 to fullfil alligment rules.
 *
 * NOTE: Due to alignment restriction and buggy drivers, do not try to use mat3 inside structs.
 * Do not use arrays of float. They are padded to arrays of vec4 and are not worth it.
 *
 * IMPORTANT: Don't forget to align mat4 and vec4 to 16 bytes.
 **/

#ifndef __cplusplus /* GLSL */
#  pragma BLENDER_REQUIRE(common_math_lib.glsl)
#  define BLI_STATIC_ASSERT_ALIGN(type_, align_)
#  define BLI_STATIC_ASSERT_SIZE(type_, size_)
#  define static
#  define inline
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

#  include "BLI_float2.hh"
#  include "BLI_float3.hh"
#  include "BLI_float4.hh"
#  include "BLI_float4x4.hh"
#  include "BLI_int2.hh"
#  include "BLI_int3.hh"

typedef float mat4[4][4];
using vec4 = blender::float4;
using vec3 = blender::float3;
using vec2 = blender::float2;
using ivec3 = blender::int3;
using ivec2 = blender::int2;
typedef uint uvec4[4];
typedef uint uvec3[3];
typedef uint uvec2[2];
typedef int bvec4[4];
typedef int bvec2[2];
/* Ugly but it does the job! */
#  define bool int

#  include "eevee_wrapper.hh"

namespace blender::eevee {

#endif

#define UBO_MIN_MAX_SUPPORTED_SIZE 1 << 14

/* -------------------------------------------------------------------- */
/** \name Sampling
 * \{ */

enum eSamplingDimension : uint32_t {
  SAMPLING_FILTER_U = 0u,
  SAMPLING_FILTER_V = 1u,
  SAMPLING_LENS_U = 2u,
  SAMPLING_LENS_V = 3u,
  SAMPLING_TIME = 4u,
  SAMPLING_SHADOW_U = 5u,
  SAMPLING_SHADOW_V = 6u,
  SAMPLING_SHADOW_W = 7u,
  SAMPLING_SHADOW_X = 8u,
  SAMPLING_SHADOW_Y = 9u,
  SAMPLING_CLOSURE = 10u,
  SAMPLING_LIGHTPROBE = 11u,
  SAMPLING_TRANSPARENCY = 12u,
  SAMPLING_SSS_U = 13u,
  SAMPLING_SSS_V = 14u
};

/** IMPORTANT: Make sure the array can contain all sampling dimensions. */
#define SAMPLING_DIMENSION_COUNT 15

struct SamplingData {
  /** Array containing random values from Low Discrepency Sequence in [0..1) range. */
  /** HACK: float arrays are padded to vec4 in GLSL. Using vec4 for now to get the same alignment
   * but this is wasteful. */
  vec4 dimensions[SAMPLING_DIMENSION_COUNT];
};
BLI_STATIC_ASSERT_ALIGN(SamplingData, 16)

/* Returns total sample count in a web pattern of the given size. */
static inline int web_sample_count_get(int web_density, int ring_count)
{
  return ((ring_count * ring_count + ring_count) / 2) * web_density + 1;
}

/* Returns lowest possible ring count that contains at least sample_count samples. */
static inline int web_ring_count_get(int web_density, int sample_count)
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

static inline bool is_panoramic(eCameraType type)
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
  FILM_DATA_DEPTH = 6u,
  FILM_DATA_MOTION = 7u
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
  /** Used for fade-in effect. */
  float opacity;
  /** Padding to sizeof(vec4). */
  int _pad0, _pad1, _pad2;
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
  float bokeh_aperture_ratio;
  /** Circle of confusion (CoC) parameters. */
  float coc_mul;
  float coc_bias;
  float coc_abs_max;
  /** Copy of camera type. */
  eCameraType camera_type;
  int _pad0, _pad1;
};
BLI_STATIC_ASSERT_ALIGN(DepthOfFieldData, 16)

static inline float coc_radius_from_camera_depth(DepthOfFieldData dof, float depth)
{
  depth = (dof.camera_type != CAMERA_ORTHO) ? 1.0f / depth : depth;
  return dof.coc_mul * depth + dof.coc_bias;
}

static inline float regular_polygon_side_length(float sides_count)
{
  return 2.0f * sinf(M_PI / sides_count);
}

/* Returns intersection ratio between the radius edge at theta and the regular polygon edge.
 * Start first corners at theta == 0. */
static inline float circle_to_polygon_radius(float sides_count, float theta)
{
  /* From Graphics Gems from CryENGINE 3 (Siggraph 2013) by Tiago Sousa (slide
   * 36). */
  float side_angle = (2.0f * M_PI) / sides_count;
  return cosf(side_angle * 0.5f) /
         cosf(theta - side_angle * floorf((sides_count * theta + M_PI) / (2.0f * M_PI)));
}

/* Remap input angle to have homogenous spacing of points along a polygon edge.
 * Expects theta to be in [0..2pi] range. */
static inline float circle_to_polygon_angle(float sides_count, float theta)
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

/* -------------------------------------------------------------------- */
/** \name VelocityModule
 * \{ */

struct VelocityObjectData {
  mat4 next_object_mat;
  mat4 prev_object_mat;
};
BLI_STATIC_ASSERT_ALIGN(VelocityObjectData, 16)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Motion Blur
 * \{ */

#define MB_TILE_DIVISOR 32

struct MotionBlurData {
  /** Motion vector lengths are clamped to this maximum. A value of 0 means effect is bypassed. */
  float blur_max;
  /** Depth scaling factor. Avoid bluring background behind moving objects. */
  float depth_scale;
  /** As the name suggests. Used to avoid a division in the sampling. */
  vec2 target_size_inv;
  /** Viewport motion blur only blurs using previous frame vectors. */
  bool is_viewport;
  int _pad0, _pad1, _pad2;
};
BLI_STATIC_ASSERT_ALIGN(MotionBlurData, 16)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Cullings
 * \{ */

/* Number of items in a culling batch. Needs to be Power of 2. */
#define CULLING_ITEM_BATCH 128
/* Maximum number of 32 bit uint stored per tile. */
#define CULLING_MAX_WORD ((CULLING_ITEM_BATCH + 1) / 32)
/* TODO(fclem) Support more than 4 words using layered texture for culling result. */
#if CULLING_MAX_WORD > 4
#  error "CULLING_MAX_WORD is greater than supported maximum."
#endif
/* Fine grained subdivision in the Z direction. */
#define CULLING_ZBIN_COUNT 4088

struct CullingData {
  /* Linearly distributed z-bins with encoded uint16_t min and max index. */
  /* NOTE: due to alignment restrictions of uint arrays, use uvec4. */
  uvec4 zbins[CULLING_ZBIN_COUNT / 4];
  /* Extent of one square tile in pixels. */
  int tile_size;
  /* Valid item count in the data array. */
  uint items_count;
  /* Scale and bias applied to linear Z to get zbin. */
  float zbin_scale;
  float zbin_bias;
  /* Scale applied to tile pixel coordinates to get target UV coordinate. */
  vec2 tile_to_uv_fac;
  vec2 _pad0;
};
BLI_STATIC_ASSERT_ALIGN(CullingData, 16)
BLI_STATIC_ASSERT_SIZE(CullingData, UBO_MIN_MAX_SUPPORTED_SIZE)

static inline int culling_z_to_zbin(CullingData data, float z)
{
  return int(z * data.zbin_scale + data.zbin_bias);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Lights
 * \{ */

#define LIGHT_NO_SHADOW -1

enum eLightType : uint32_t {
  LIGHT_SUN = 0u,
  LIGHT_POINT = 1u,
  LIGHT_SPOT = 2u,
  LIGHT_RECT = 3u,
  LIGHT_ELLIPSE = 4u
};

static inline bool is_area_light(eLightType type)
{
  return type >= LIGHT_RECT;
}

struct LightData {
  /** Normalized obmat. Last column contains data accessible using the following macros. */
  mat4 object_mat;
  /** Packed data in the last column of the object_mat. */
#define _area_size_x object_mat[0][3]
#define _area_size_y object_mat[1][3]
#define _radius _area_size_x
#define _spot_mul object_mat[2][3]
#define _spot_bias object_mat[3][3]
  /** Aliases for axes. */
#ifdef __cplusplus
#  define _right object_mat[0]
#  define _up object_mat[1]
#  define _back object_mat[2]
#  define _position object_mat[3]
#else
#  define _right object_mat[0].xyz
#  define _up object_mat[1].xyz
#  define _back object_mat[2].xyz
#  define _position object_mat[3].xyz
#endif
  /** Influence radius (inversed and squared) adjusted for Surface / Volume power. */
  float influence_radius_invsqr_surface;
  float influence_radius_invsqr_volume;
  /** Maximum influence radius. Used for culling. */
  float influence_radius_max;
  /** Offset in the shadow struct table. -1 means no shadow. */
  int shadow_id;
  /** NOTE: It is ok to use vec3 here. A float is declared right after it.
   * vec3 is also aligned to 16 bytes. */
  vec3 color;
  /** Power depending on shader type. */
  float diffuse_power;
  float specular_power;
  float volume_power;
  /** Special radius factor for point lighting. */
  float radius_squared;
  /** Light Type. */
  eLightType type;
  /** Spot size. Aligned to size of vec2. */
  vec2 spot_size_inv;
  /** Padding to sizeof(vec4). */
  float _pad0;
  float _pad1;
};
BLI_STATIC_ASSERT_ALIGN(LightData, 16)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shadows
 * \{ */

/**
 * A point light shadow is composed of 1, 5 or 6 shadow regions.
 * Regions are sorted in this order -Z, +X, -X, +Y, -Y, +Z.
 * Face index is computed from light's object space coordinates.
 */
struct ShadowPunctualData {
  /** Shadow matrix to convert Local face coordinates to UV space [0..1]. */
  mat4 shadow_mat;
  /** NOTE: It is ok to use vec3 here. A float is declared right after it.
   * vec3 is also aligned to 16 bytes. */
  /** Shadow offset caused by jittering projection origin (for soft shadows). */
  vec3 shadow_offset;
  /** Shadow bias in world space. */
  float shadow_bias;
  /** Offset from the first region to the second one. All regions are stored vertically. */
  float region_offset;
  /** True if shadow is omnidirectional and there is 6 fullsized shadow regions.  */
  bool is_omni;
  /** Padding to sizeof(vec4). */
  int _pad0;
  int _pad1;
};
BLI_STATIC_ASSERT_ALIGN(ShadowPunctualData, 16)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Light Probes
 * \{ */

/**
 * Data used when filtering the cubemap.
 * NOTE(fclem): We might want to promote some of theses to push constants as they are changed
 * very frequently (Vulkan).
 */
struct LightProbeFilterData {
  /** For glossy filter. */
  float roughness;
  /** Higher bias lowers the noise but increases blur and reduces quality. */
  float lod_bias;
  /** Final intensity multiplicator. */
  float instensity_fac;
  /** Luma maximum value. */
  float luma_max;
  /** Sample count to take from the input cubemap. */
  float sample_count;
  /** Visibility blur ratio [0..1]. Converted to angle in [0..PI/2] range. */
  float visibility_blur;
  /** Depth range to encode in the resulting visibility map. */
  float visibility_range;
  /** Target layer to render the fullscreen triangle to. */
  int target_layer;
};
BLI_STATIC_ASSERT_ALIGN(LightProbeFilterData, 16)

/**
 * Common data to all irradiance grid.
 */
struct GridInfoData {
  mat4 lookdev_rotation;
  /** Total of visibility cells per row and layer. */
  int visibility_cells_per_row;
  int visibility_cells_per_layer;
  /** Size of visibility cell. */
  int visibility_size;
  /** Number of irradiance cells per row. */
  int irradiance_cells_per_row;
  /** Smooth irradiance sample interpolation but increases light leaks. */
  float irradiance_smooth;
  /** Total number of active irradiance grid including world. */
  int grid_count;
  /** Display size of sample spheres. */
  float display_size;
  float _pad0;
};
BLI_STATIC_ASSERT_ALIGN(GridInfoData, 16)

/**
 * Data for a single irradiance grid.
 */
struct GridData {
  /** Object matrix inverse (World -> Local). */
  mat4 local_mat;
  /** Resolution of the light grid. */
  ivec3 resolution;
  /** Offset of the first cell of this grid in the grid texture. */
  int offset;
  /** World space vector between 2 adjacent cells. */
  vec3 increment_x;
  /** Attenuation Bias. */
  float attenuation_bias;
  /** World space vector between 2 adjacent cells. */
  vec3 increment_y;
  /** Attenuation scaling. */
  float attenuation_scale;
  /** World space vector between 2 adjacent cells. */
  vec3 increment_z;
  /** Number of grid levels not ready for display during baking. */
  int level_skip;
  /** World space corner position. */
  vec3 corner;
  /** Visibility test parameters. */
  float visibility_range;
  float visibility_bleed;
  float visibility_bias;
  float _pad0;
  float _pad1;
};
BLI_STATIC_ASSERT_ALIGN(GridData, 16)

static inline ivec3 grid_cell_index_to_coordinate(int cell_id, ivec3 resolution)
{
  ivec3 cell_coord;
  cell_coord.z = cell_id % resolution.z;
  cell_coord.y = (cell_id / resolution.z) % resolution.y;
  cell_coord.x = cell_id / (resolution.z * resolution.y);
  return cell_coord;
}

/**
 * Common data to all cubemaps.
 */
struct CubemapInfoData {
  mat4 lookdev_rotation;
  /** LOD containing data for roughness of 1. */
  float roughness_max_lod;
  /** Total number of active cubemaps including world. */
  int cube_count;
  /** Display size of sample spheres. */
  float display_size;
  float _pad2;
};
BLI_STATIC_ASSERT_ALIGN(CubemapInfoData, 16)

#define CUBEMAP_SHAPE_SPHERE 0.0
#define CUBEMAP_SHAPE_BOX 1.0

/**
 * Data for a single reflection cubemap probe.
 */
struct CubemapData {
  /** Influence shape matrix (World -> Local). */
  mat4 influence_mat;
  /** Packed data in the last column of the influence_mat. */
#define _attenuation_factor influence_mat[0][3]
#define _attenuation_type influence_mat[1][3]
#define _parallax_type influence_mat[2][3]
  /** Layer of the cube array to sample. */
#define _layer influence_mat[3][3]
  /** Parallax shape matrix (World -> Local). */
  mat4 parallax_mat;
  /** Packed data in the last column of the parallax_mat. */
#define _world_position_x parallax_mat[0][3]
#define _world_position_y parallax_mat[1][3]
#define _world_position_z parallax_mat[2][3]
};
BLI_STATIC_ASSERT_ALIGN(CubemapData, 16)

struct LightProbeInfoData {
  GridInfoData grids;
  CubemapInfoData cubes;
};
BLI_STATIC_ASSERT_ALIGN(LightProbeInfoData, 16)

#define GRID_MAX 64

/** \} */

/* -------------------------------------------------------------------- */
/** \name Subsurface
 * \{ */

#define SSS_SAMPLE_MAX 64
#define BURLEY_TRUNCATE 16.0
#define BURLEY_TRUNCATE_CDF 0.9963790093708328

struct SubsurfaceData {
  /** xy: 2D sample position [-1..1], zw: sample_bounds. */
  /* NOTE(fclem) Using vec4 for alignment. */
  vec4 samples[SSS_SAMPLE_MAX];
  /** Sample index after which samples are not randomly rotated anymore. */
  int jitter_threshold;
  /** Number of samples precomputed in the set. */
  int sample_len;
  int _pad0;
  int _pad1;
};
BLI_STATIC_ASSERT_ALIGN(SubsurfaceData, 16)

/** \} */

/* -------------------------------------------------------------------- */
/** \name Utility Texture
 * \{ */

#define UTIL_TEX_SIZE 64
#define UTIL_BTDF_LAYER_COUNT 16
/* Scale and bias to avoid interpolation of the border pixel.
 * Remap UVs to the border pixels centers. */
#define UTIL_TEX_UV_SCALE ((UTIL_TEX_SIZE - 1.0f) / UTIL_TEX_SIZE)
#define UTIL_TEX_UV_BIAS (0.5f / UTIL_TEX_SIZE)

#define UTIL_BLUE_NOISE_LAYER 0
#define UTIL_LTC_MAT_LAYER 1
#define UTIL_LTC_MAG_LAYER 2
#define UTIL_BSDF_LAYER 2
#define UTIL_BTDF_LAYER 3
#define UTIL_DISK_INTEGRAL_LAYER 3
#define UTIL_DISK_INTEGRAL_COMP 2

#ifndef __cplusplus
/* For codestyle reasons, we do not declare samplers in lib files. Use a prototype instead. */
vec4 utility_tx_fetch(vec2 texel, float layer);
vec4 utility_tx_sample(vec2 uv, float layer);

/* Fetch texel. Wrapping if above range. */
#  define utility_tx_fetch_define(utility_tx_) \
    vec4 utility_tx_fetch(vec2 texel, float layer) \
    { \
      return texelFetch(utility_tx_, ivec3(ivec2(texel) % UTIL_TEX_SIZE, layer), 0); \
    }

/* Sample at uv position. Filtered & Wrapping enabled. */
#  define utility_tx_sample_define(utility_tx_) \
    vec4 utility_tx_sample(vec2 uv, float layer) \
    { \
      return textureLod(utility_tx_, vec3(uv, layer), 0.0); \
    }

/* Stubs declarations if not using it. */
#  define utility_tx_fetch_define_stub(utility_tx_) \
    vec4 utility_tx_fetch(vec2 texel, float layer) \
    { \
      return vec4(0); \
    }
#  define utility_tx_sample_define_stub(utility_tx_) \
    vec4 utility_tx_sample(vec2 uv, float layer) \
    { \
      return vec4(0); \
    }
#endif

/** \} */

#ifdef __cplusplus
using CameraDataBuf = StructBuffer<CameraData>;
using CubemapDataBuf = StructArrayBuffer<CubemapData, CULLING_ITEM_BATCH>;
using CullingDataBuf = StructBuffer<CullingData>;
using DepthOfFieldDataBuf = StructBuffer<DepthOfFieldData>;
using GridDataBuf = StructArrayBuffer<GridData, GRID_MAX>;
using LightDataBuf = StructArrayBuffer<LightData, CULLING_ITEM_BATCH>;
using LightProbeFilterDataBuf = StructBuffer<LightProbeFilterData>;
using LightProbeInfoDataBuf = StructBuffer<LightProbeInfoData>;
using ShadowPunctualDataBuf = StructArrayBuffer<ShadowPunctualData, CULLING_ITEM_BATCH>;
using SubsurfaceDataBuf = StructBuffer<SubsurfaceData>;
using VelocityObjectBuf = StructBuffer<VelocityObjectData>;

#  undef bool
}  // namespace blender::eevee
#endif
