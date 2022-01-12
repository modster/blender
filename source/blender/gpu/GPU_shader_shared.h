
#ifndef USE_GPU_SHADER_CREATE_INFO
#  include "intern/gpu_shader_shared_utils.h"
#endif

#ifdef __cplusplus
using blender::float2;
using blender::float4;
using blender::float4x4;
#endif

struct NodeLinkData {
  float4 colors[3];
  float2 bezierPts[4];
  bool1 doArrow;
  bool1 doMuted;
  float dim_factor;
  float thickness;
  float dash_factor;
  float dash_alpha;
  float expandSize;
  float arrowSize;
};

struct NodeLinkInstanceData {
  float4 colors[6];
  float expandSize;
  float arrowSize;
  float2 pad;
};

struct GPencilStrokeData {
  float2 viewport;
  float pixsize;
  float objscale;
  float pixfactor;
  int xraymode;
  int caps_start;
  int caps_end;
  bool1 keep_size;
  bool1 fill_stroke;
  float2 pad;
};

struct GPUClipPlanes {
  float4x4 ModelMatrix;
  float4 world[6];
};