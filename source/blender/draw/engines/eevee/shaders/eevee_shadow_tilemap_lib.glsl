
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

/** Decoded tile data structure. */
struct ShadowTileData {
  /** Page inside the virtual shadow map atlas. */
  uvec2 page;
  /** If not 0, offset to the tilemap that has a valid page for this position. */
  uint lod_tilemap_offset;
  /** Set to true during the setup phase if the tile is inside the view frustum. */
  bool is_visible;
  /** If the tile is needed for rendering. */
  bool is_used;
  /** True if an update is needed. */
  bool do_update;
};

#define SHADOW_TILE_NO_DATA 0u
#define SHADOW_TILE_DO_UPDATE (1u << 29u)
#define SHADOW_TILE_IS_VISIBLE (1u << 30u)
#define SHADOW_TILE_IS_USED (1u << 31u)

ShadowTileData shadow_tile_data_unpack(uint data)
{
  ShadowTileData tile;
  tile.page.x = data & 0xFu;
  tile.page.y = (data >> 4u) & 0xFu;
  tile.lod_tilemap_offset = (data >> 8u) & 0xFu;
  tile.is_visible = flag_test(data, SHADOW_TILE_IS_VISIBLE);
  tile.is_used = flag_test(data, SHADOW_TILE_IS_USED);
  tile.do_update = flag_test(data, SHADOW_TILE_DO_UPDATE);
  return tile;
}

uint shadow_tile_data_pack(ShadowTileData tile)
{
  uint data;
  data = tile.page.x;
  data |= tile.page.y << 4u;
  data |= tile.lod_tilemap_offset << 8u;
  set_flag_from_test(data, tile.is_visible, SHADOW_TILE_IS_VISIBLE);
  set_flag_from_test(data, tile.is_used, SHADOW_TILE_IS_USED);
  set_flag_from_test(data, tile.do_update, SHADOW_TILE_DO_UPDATE);
  return data;
}

int shadow_tile_index(ivec2 tile)
{
  return tile.x + tile.y * SHADOW_TILEMAP_RES;
}

ivec2 shadow_tile_coord(int tile_index)
{
  return ivec2(tile_index % SHADOW_TILEMAP_RES, tile_index / SHADOW_TILEMAP_RES);
}

/* Return bottom left pixel position of the tilemap inside the tilemap atlas. */
ivec2 shadow_tilemap_start(int tilemap_index)
{
  return SHADOW_TILEMAP_RES *
         ivec2(tilemap_index % SHADOW_TILEMAP_PER_ROW, tilemap_index / SHADOW_TILEMAP_PER_ROW);
}

ivec2 shadow_tile_coord_in_atlas(ivec2 tile, int tilemap_index)
{
  return shadow_tilemap_start(tilemap_index) + tile;
}

void shadow_tile_store(restrict uimage2D tilemaps_img,
                       ivec2 tile_co,
                       int tilemap_index,
                       ShadowTileData data)
{
  uint tile_data = shadow_tile_data_pack(data);
  imageStore(tilemaps_img, shadow_tile_coord_in_atlas(tile_co, tilemap_index), uvec4(tile_data));
}
/* Ugly define because some compilers seems to not like the fact the imageAtomicOr is inside
 * a function. */
#define shadow_tile_set_flag(tilemaps_img, tile_co, tilemap_index, flag) \
  imageAtomicOr(tilemaps_img, shadow_tile_coord_in_atlas(tile_co, tilemap_index), flag)

ShadowTileData shadow_tile_load(restrict uimage2D tilemaps_img, ivec2 tile_co, int tilemap_index)
{
  uint tile_data = SHADOW_TILE_NO_DATA;
  if (in_range_inclusive(tile_co, ivec2(0), ivec2(SHADOW_TILEMAP_RES - 1))) {
    tile_data = imageLoad(tilemaps_img, shadow_tile_coord_in_atlas(tile_co, tilemap_index)).x;
  }
  return shadow_tile_data_unpack(tile_data);
}

ShadowTileData shadow_tile_load(usampler2D tilemaps_tx, ivec2 tile_co, int tilemap_index)
{
  uint tile_data = SHADOW_TILE_NO_DATA;
  if (in_range_inclusive(tile_co, ivec2(0), ivec2(SHADOW_TILEMAP_RES - 1))) {
    tile_data = texelFetch(tilemaps_tx, shadow_tile_coord_in_atlas(tile_co, tilemap_index), 0).x;
  }
  return shadow_tile_data_unpack(tile_data);
}

/* Return the correct tilemap index given a world space position. */
int shadow_directional_tilemap_index(ShadowData shadow, vec3 P)
{
  vec3 shadow_map_center = shadow.mat[3].xyz;
  float dist_to_center = distance(shadow_map_center, P);
  float clipmap_level = log2(dist_to_center);
  /* Use floor because we can have negative numbers. */
  int clipmap_lod = int(floor(clipmap_level));
  clipmap_lod = clamp(clipmap_lod, shadow.clipmap_lod_min, shadow.clipmap_lod_max);
  return shadow.tilemap_index + clipmap_lod - shadow.clipmap_lod_min;
}
