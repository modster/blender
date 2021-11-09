
/**
 * Debug drawing for virtual shadowmaps.
 * See eShadowDebug for more information.
 */

#pragma BLENDER_REQUIRE(common_view_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_light_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_sampling_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shadow_tilemap_lib.glsl)

/** Control the scaling of the tilemap splat. */
const float pixel_scale = 5.0;

layout(std140) uniform debug_block
{
  ShadowDebugData debug;
};

layout(std430, binding = 0) readonly buffer tilemaps_block
{
  ShadowTileMapData tilemaps[];
};

uniform usampler2D tilemaps_tx;
uniform sampler2D depth_tx;

in vec4 uvcoordsvar;

layout(location = 0, index = 0) out vec4 out_color_add;
layout(location = 0, index = 1) out vec4 out_color_mul;

vec3 debug_random_color(ivec2 v)
{
  float r = interlieved_gradient_noise(vec2(v), 0.0, 0.0);
  return hue_gradient(r);
}

vec3 debug_random_color(int v)
{
  return debug_random_color(ivec2(v, 0));
}

vec3 debug_tile_state_color(ShadowTileData tile)
{
  if (tile.do_update && tile.is_used && tile.is_visible) {
    return vec3(1, 0, 0);
  }
  else if (tile.is_used && tile.is_visible) {
    return vec3(0, 1, 0);
  }
  else if (tile.is_visible) {
    return vec3(0, 0.2, 0.8);
  }
  return vec3(0);
}

bool debug_tilemap()
{
  ivec2 tile = ivec2(gl_FragCoord.xy / pixel_scale);
  int tilemap_lod = tile.x / (SHADOW_TILEMAP_RES + 2);
  int tilemap_index = tile.y / (SHADOW_TILEMAP_RES + 2);
  tile = (tile % (SHADOW_TILEMAP_RES + 2)) - 1;
  tilemap_index += debug.shadow.tilemap_index;

  if ((tilemap_index >= debug.shadow.tilemap_index) &&
      (tilemap_index <= debug.shadow.tilemap_last) && (tilemap_lod == 0) &&
      in_range_inclusive(tile, ivec2(0), ivec2(SHADOW_TILEMAP_RES - 1))) {
    ShadowTileData tile_data = shadow_tile_load(tilemaps_tx, tile, tilemap_index);
    /* Write depth to overlap overlays. */
    gl_FragDepth = 0.0;
    out_color_add = vec4(debug_tile_state_color(tile_data), 0);
    out_color_mul = vec4(0);
    return true;
  }
  return false;
}

bool debug_tilemap_point_is_inside(vec3 P, int tilemap_index)
{
  int tilemap_data_index = debug.tilemap_data_index + tilemap_index - debug.shadow.tilemap_index;
  vec3 clipP = project_point(tilemaps[tilemap_data_index].persmat, P);
  return in_range_inclusive(clipP, vec3(-1.0), vec3(1.0));
}

/** Unlike shadow_directional_tilemap_index, returns the first tilemap overlapping the position. */
int debug_directional_tilemap_index(vec3 P)
{
  for (int tilemap_index = debug.shadow.tilemap_index; tilemap_index <= debug.shadow.tilemap_last;
       tilemap_index++) {
    if (debug_tilemap_point_is_inside(P, tilemap_index)) {
      return tilemap_index;
    }
  }
  return -1;
}

int debug_punctual_tilemap_index(vec3 P)
{
  vec3 L;
  float dist;
  light_vector_get(debug.light, P, L, dist);
  vec3 lL = light_world_to_local(debug.light, -L) * dist;
  lL -= debug.shadow.offset;
  int tilemap_index = debug.shadow.tilemap_index + shadow_punctual_face_index_get(lL);
  if (tilemap_index > debug.shadow.tilemap_last) {
    return -1;
  }
  if (debug_tilemap_point_is_inside(P, tilemap_index)) {
    return tilemap_index;
  }
  return -1;
}

void debug_lod(vec3 P)
{
  int tilemap_index = (debug.light.type == LIGHT_SUN) ? debug_directional_tilemap_index(P) :
                                                        debug_punctual_tilemap_index(P);
  if (tilemap_index != -1) {
    vec3 color = debug_random_color(tilemap_index);
    out_color_add = vec4(color * 0.5, 0.0);
    out_color_mul = out_color_add;
  }
  else {
    out_color_add = vec4(0.0);
    out_color_mul = vec4(0.0);
  }
}

void debug_tile_state(vec3 P)
{
  int tilemap_index = (debug.light.type == LIGHT_SUN) ? debug_directional_tilemap_index(P) :
                                                        debug_punctual_tilemap_index(P);
  if (tilemap_index != -1) {
    int tilemap_data_index = debug.tilemap_data_index + tilemap_index - debug.shadow.tilemap_index;
    vec3 clipP = project_point(tilemaps[tilemap_data_index].persmat, P);
    ivec2 tile = ivec2((clipP * 0.5 + 0.5) * SHADOW_TILEMAP_RES);
    ShadowTileData tile_data = shadow_tile_load(tilemaps_tx, tile, tilemap_index);
    vec3 color = debug_tile_state_color(tile_data);
    out_color_add = vec4(color * 0.5, 0);
    out_color_mul = out_color_add;
  }
  else {
    out_color_add = vec4(0.0);
    out_color_mul = vec4(0.0);
  }
}

void main()
{
  /* Default to no output. */
  gl_FragDepth = 1.0;
  out_color_add = vec4(0.0);
  out_color_mul = vec4(1.0);

  if (debug_tilemap()) {
    return;
  }

  float depth = texelFetch(depth_tx, ivec2(gl_FragCoord.xy), 0).r;
  vec3 P = get_world_space_from_depth(uvcoordsvar.xy, depth);
  /* Make it pass the depth test. */
  gl_FragDepth = depth - 1e-6;

  if (depth != 1.0) {
    switch (debug.type) {
      case SHADOW_DEBUG_TILEMAPS:
        debug_tile_state(P);
        break;
      case SHADOW_DEBUG_LOD:
        debug_lod(P);
        break;
      default:
        discard;
    }
  }
}