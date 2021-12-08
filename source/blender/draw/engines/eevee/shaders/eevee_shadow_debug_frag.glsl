
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
#pragma BLENDER_REQUIRE(eevee_shadow_page_lib.glsl)

/** Control the scaling of the tilemap splat. */
const float pixel_scale = 4.0;

layout(std140) uniform debug_block
{
  ShadowDebugData debug;
};

layout(std430, binding = 0) readonly buffer tilemaps_buf
{
  ShadowTileMapData tilemaps[];
};

layout(std430, binding = 1) readonly buffer pages_buf
{
  ShadowPagePacked pages[];
};

uniform usampler2D tilemaps_tx;
uniform sampler2D depth_tx;
uniform sampler2D atlas_tx;

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
  if (tile.lod > 0) {
    return vec3(1, 0.5, 0) * float(tile.lod) / float(SHADOW_TILEMAP_LOD);
  }
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
  int tilemap_lod = tile.y / (SHADOW_TILEMAP_RES + 2);
  int tilemap_index = tile.x / (SHADOW_TILEMAP_RES + 2);
  tile = (tile % (SHADOW_TILEMAP_RES + 2)) - 1;
  tilemap_index += debug.shadow.tilemap_index;
  int tilemap_lod_max = (debug.light.type != LIGHT_SUN) ? SHADOW_TILEMAP_LOD : 0;

  if ((tilemap_index >= debug.shadow.tilemap_index) &&
      (tilemap_index <= debug.shadow.tilemap_last) && (tilemap_lod >= 0) &&
      (tilemap_lod <= tilemap_lod_max) &&
      in_range_inclusive(tile, ivec2(0), ivec2(SHADOW_TILEMAP_RES - 1))) {
    tile >>= tilemap_lod;
    ShadowTileData tile_data = shadow_tile_load(tilemaps_tx, tile, tilemap_lod, tilemap_index);
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
  vec3 clipP = project_point(tilemaps[tilemap_data_index].tilemat, P);
  return in_range_inclusive(clipP, vec3(0.0), vec3(SHADOW_TILEMAP_RES));
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

void debug_pages(vec3 P)
{
  int tilemap_index = (debug.light.type == LIGHT_SUN) ? debug_directional_tilemap_index(P) :
                                                        debug_punctual_tilemap_index(P);
  if (tilemap_index != -1) {
    int tilemap_data_index = debug.tilemap_data_index + tilemap_index - debug.shadow.tilemap_index;
    vec3 clipP = project_point(tilemaps[tilemap_data_index].tilemat, P);
    ivec2 tile = ivec2(clipP.xy);
    ShadowTileData tile_data = shadow_tile_load(tilemaps_tx, tile, 0, tilemap_index);
    vec3 color = debug_random_color(ivec2(tile_data.page));
    out_color_add = vec4(color * 0.5, 0);
    out_color_mul = out_color_add * 0.5 + 0.5;
  }
  else {
    out_color_add = vec4(0.0);
    out_color_mul = vec4(0.5);
  }
}

void debug_lod(vec3 P)
{
  int tilemap_index = (debug.light.type == LIGHT_SUN) ? debug_directional_tilemap_index(P) :
                                                        debug_punctual_tilemap_index(P);
  if (tilemap_index != -1) {
    vec3 color = debug_random_color(tilemap_index);
    out_color_add = vec4(color * 0.5, 0.0);
    out_color_mul = out_color_add * 0.5 + 0.5;
  }
  else {
    out_color_add = vec4(0.0);
    out_color_mul = vec4(0.5);
  }
}

void debug_tile_state(vec3 P)
{
  int tilemap_index = (debug.light.type == LIGHT_SUN) ? debug_directional_tilemap_index(P) :
                                                        debug_punctual_tilemap_index(P);
  if (tilemap_index != -1) {
    int tilemap_data_index = debug.tilemap_data_index + tilemap_index - debug.shadow.tilemap_index;
    vec3 clipP = project_point(tilemaps[tilemap_data_index].tilemat, P);
    ivec2 tile = ivec2(clipP.xy);
    ShadowTileData tile_data = shadow_tile_load(tilemaps_tx, tile, 0, tilemap_index);
    vec3 color = debug_tile_state_color(tile_data);
    out_color_add = vec4(color * 0.5, 0);
    out_color_mul = out_color_add * 0.5 + 0.5;
  }
  else {
    out_color_add = vec4(0.0);
    out_color_mul = vec4(0.5);
  }
}

void debug_page_allocation(void)
{
  ivec2 page = ivec2(gl_FragCoord.xy / pixel_scale);

  if (in_range_inclusive(page, ivec2(0), ivec2(SHADOW_PAGE_PER_ROW - 1))) {
    uint page_index = shadow_page_to_index(page);
    if (pages[page_index] != SHADOW_PAGE_NO_DATA) {
      out_color_add = vec4(1, 1, 1, 0);
    }
    else {
      out_color_add = vec4(0, 0, 0, 0);
    }
    out_color_mul = vec4(0);
    /* Write depth to overlap overlays. */
    gl_FragDepth = 0.0;
  }
}

void debug_tile_allocation(void)
{
  ivec2 tile_co = ivec2(gl_FragCoord.xy) - 32;
  /* Assumes tilemap buffer is squared. */
  if (in_range_inclusive(tile_co, ivec2(0), textureSize(tilemaps_tx, 0).xy - 1)) {
    ShadowTileData tile = shadow_tile_data_unpack(texelFetch(tilemaps_tx, tile_co, 0).x);
    out_color_add = vec4(debug_tile_state_color(tile), 0);
    out_color_mul = vec4(0);
    /* Write depth to overlap overlays. */
    gl_FragDepth = 0.0;
  }
}

void debug_shadow_depth(vec3 P)
{
  vec3 L;
  float dist;
  light_vector_get(debug.light, P, L, dist);
  vec3 lL = light_world_to_local(debug.light, -L) * dist;
  lL -= debug.shadow.offset;
  vec3 lP = transform_point(debug.shadow.mat, P);
  float depth;
  if (debug.light.type == LIGHT_SUN) {
    shadow_directional_depth_get(
        atlas_tx, tilemaps_tx, debug.light, debug.shadow, debug.camera_position, lP, P);
  }
  else {
    shadow_punctual_depth_get(atlas_tx, tilemaps_tx, debug.light, debug.shadow, lL);
  }
  out_color_add = vec4(vec3(depth), 0);
  out_color_mul = vec4(0);
}

void main()
{
  /* Default to no output. */
  gl_FragDepth = 1.0;
  out_color_add = vec4(0.0);
  out_color_mul = vec4(1.0);

  if (debug.type == SHADOW_DEBUG_PAGE_ALLOCATION) {
    debug_page_allocation();
    return;
  }

  if (debug.type == SHADOW_DEBUG_TILE_ALLOCATION) {
    debug_tile_allocation();
    return;
  }

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
      case SHADOW_DEBUG_PAGES:
        debug_pages(P);
        break;
      case SHADOW_DEBUG_LOD:
        debug_lod(P);
        break;
      case SHADOW_DEBUG_SHADOW_DEPTH:
        debug_shadow_depth(P);
        break;
      default:
        discard;
    }
  }
}