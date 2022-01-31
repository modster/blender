
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(eevee_shader_shared.hh)

/* ---------------------------------------------------------------------- */
/** \name Tilemap data
 * \{ */

/** Decoded tile data structure. */
struct ShadowTileData {
  /** Page inside the virtual shadow map atlas. */
  uvec2 page;
  /** Page owner index inside free_page_owners heap. Only valid if is_cached is true. */
  uint free_page_owner_index;
  /** Lod pointed to by LOD 0 tile page. (cubemap only) */
  uint lod;
  /** Set to true during the setup phase if the tile is inside the view frustum. */
  bool is_visible;
  /** If the tile is needed for rendering. */
  bool is_used;
  /** True if this tile owns the page and if it points to a valid page. */
  bool is_allocated;
  /** True if an update is needed. */
  bool do_update;
  /** True if the tile is indexed inside the free_page_owners heap. */
  bool is_cached;
};

#define SHADOW_TILE_NO_DATA 0u
#define SHADOW_TILE_IS_CACHED (1u << 27u)
#define SHADOW_TILE_IS_ALLOCATED (1u << 28u)
#define SHADOW_TILE_DO_UPDATE (1u << 29u)
#define SHADOW_TILE_IS_VISIBLE (1u << 30u)
#define SHADOW_TILE_IS_USED (1u << 31u)

ShadowTileData shadow_tile_data_unpack(uint data)
{
  ShadowTileData tile;
  /* Tweaked for SHADOW_PAGE_PER_ROW = 64. */
  tile.page.x = data & 63u;
  tile.page.y = (data >> 6u) & 63u;
  /* Tweaked for SHADOW_TILEMAP_LOD < 8. */
  tile.lod = (data >> 12u) & 7u;
  /* Tweaked for SHADOW_MAX_TILEMAP = 4096. */
  tile.free_page_owner_index = (data >> 15u) & 4095u;
  tile.is_visible = flag_test(data, SHADOW_TILE_IS_VISIBLE);
  tile.is_used = flag_test(data, SHADOW_TILE_IS_USED);
  tile.is_cached = flag_test(data, SHADOW_TILE_IS_CACHED);
  tile.is_allocated = flag_test(data, SHADOW_TILE_IS_ALLOCATED);
  tile.do_update = flag_test(data, SHADOW_TILE_DO_UPDATE);
  return tile;
}

uint shadow_tile_data_pack(ShadowTileData tile)
{
  uint data;
  data = tile.page.x;
  data |= tile.page.y << 6u;
  data |= tile.lod << 12u;
  data |= tile.free_page_owner_index << 15u;
  set_flag_from_test(data, tile.is_visible, SHADOW_TILE_IS_VISIBLE);
  set_flag_from_test(data, tile.is_used, SHADOW_TILE_IS_USED);
  set_flag_from_test(data, tile.is_allocated, SHADOW_TILE_IS_ALLOCATED);
  set_flag_from_test(data, tile.is_cached, SHADOW_TILE_IS_CACHED);
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
  /* Assumes base map is squared. */
  ivec2 start = SHADOW_TILEMAP_RES * ivec2(tilemap_index % SHADOW_TILEMAP_PER_ROW,
                                           tilemap_index / SHADOW_TILEMAP_PER_ROW);
  return start;
}

/* Return bottom left pixel position of the tilemap inside the tilemap atlas. */
ivec2 shadow_tilemap_start(int tilemap_index, int lod)
{
  /* Assumes base map is squared. */
  ivec2 start = shadow_tilemap_start(tilemap_index) >> lod;
  if (lod > 0) {
    const int lod0_res = SHADOW_TILEMAP_RES * SHADOW_TILEMAP_PER_ROW;
    start.y += lod0_res;
    start.x += lod0_res - (lod0_res >> (lod - 1));
  }
  return start;
}

ivec2 shadow_tile_coord_in_atlas(ivec2 tile, int tilemap_index)
{
  return shadow_tilemap_start(tilemap_index) + tile;
}

ivec2 shadow_tile_coord_in_atlas(ivec2 tile, int tilemap_index, int lod)
{
  return shadow_tilemap_start(tilemap_index, lod) + tile;
}

/** \} */

/* ---------------------------------------------------------------------- */
/** \name Load / Store functions.
 * \{ */

void shadow_tile_store(restrict uimage2D tilemaps_img,
                       ivec2 tile_co,
                       int tilemap_index,
                       ShadowTileData data)
{
  uint tile_data = shadow_tile_data_pack(data);
  imageStore(tilemaps_img, shadow_tile_coord_in_atlas(tile_co, tilemap_index), uvec4(tile_data));
}

void shadow_tile_store(
    restrict uimage2D tilemaps_img, ivec2 tile_co, int lod, int tilemap_index, ShadowTileData data)
{
  uint tile_data = shadow_tile_data_pack(data);
  imageStore(
      tilemaps_img, shadow_tile_coord_in_atlas(tile_co, tilemap_index, lod), uvec4(tile_data));
}
/* Ugly define because some compilers seems to not like the fact the imageAtomicOr is inside
 * a function. */
#define shadow_tile_set_flag(tilemaps_img, tile_co, lod, tilemap_index, flag) \
  imageAtomicOr(tilemaps_img, shadow_tile_coord_in_atlas(tile_co, tilemap_index, lod), flag)
#define shadow_tile_unset_flag(tilemaps_img, tile_co, lod, tilemap_index, flag) \
  imageAtomicAnd(tilemaps_img, shadow_tile_coord_in_atlas(tile_co, tilemap_index, lod), ~(flag))

ShadowTileData shadow_tile_load(restrict uimage2D tilemaps_img,
                                ivec2 tile_co,
                                int lod,
                                int tilemap_index)
{
  uint tile_data = SHADOW_TILE_NO_DATA;
  if (in_range_inclusive(tile_co, ivec2(0), ivec2(SHADOW_TILEMAP_RES - 1))) {
    tile_data = imageLoad(tilemaps_img, shadow_tile_coord_in_atlas(tile_co, tilemap_index, lod)).x;
  }
  return shadow_tile_data_unpack(tile_data);
}

ShadowTileData shadow_tile_load(usampler2D tilemaps_tx, ivec2 tile_co, int lod, int tilemap_index)
{
  uint tile_data = SHADOW_TILE_NO_DATA;
  if (in_range_inclusive(tile_co, ivec2(0), ivec2(SHADOW_TILEMAP_RES - 1))) {
    tile_data =
        texelFetch(tilemaps_tx, shadow_tile_coord_in_atlas(tile_co, tilemap_index, lod), 0).x;
  }
  return shadow_tile_data_unpack(tile_data);
}

/* This function should be the inverse of ShadowTileMap::tilemap_coverage_get. */
int shadow_directional_clipmap_level(ShadowData shadow, float distance_to_camera)
{
  /* Why do we need to bias by 2 here? I don't know... */
  int clipmap_lod = int(ceil(log2(distance_to_camera))) + 2;
  return clamp(clipmap_lod, shadow.clipmap_lod_min, shadow.clipmap_lod_max);
}

/** \} */

/* ---------------------------------------------------------------------- */
/** \name Frustum shapes.
 * \{ */

vec3 shadow_tile_corner_persp(ShadowTileMapData tilemap, ivec2 tile)
{
  return tilemap.corners[1].xyz + tilemap.corners[2].xyz * float(tile.x) +
         tilemap.corners[3].xyz * float(tile.y);
}

Pyramid shadow_tilemap_cubeface_bounds(ShadowTileMapData tilemap,
                                       ivec2 tile_start,
                                       const ivec2 extent)
{
  Pyramid shape;
  shape.corners[0] = tilemap.corners[0].xyz;
  shape.corners[1] = shadow_tile_corner_persp(tilemap, tile_start + ivec2(0, 0));
  shape.corners[2] = shadow_tile_corner_persp(tilemap, tile_start + ivec2(extent.x, 0));
  shape.corners[3] = shadow_tile_corner_persp(tilemap, tile_start + extent);
  shape.corners[4] = shadow_tile_corner_persp(tilemap, tile_start + ivec2(0, extent.y));
  return shape;
}

vec3 shadow_tile_corner_ortho(ShadowTileMapData tilemap, ivec2 tile, const bool far)
{
  return tilemap.corners[0].xyz + tilemap.corners[1].xyz * float(tile.x) +
         tilemap.corners[2].xyz * float(tile.y) + tilemap.corners[3].xyz * float(far);
}

Box shadow_tilemap_clipmap_bounds(ShadowTileMapData tilemap, ivec2 tile_start, const ivec2 extent)
{
  Box shape;
  shape.corners[0] = shadow_tile_corner_ortho(tilemap, tile_start + ivec2(0, 0), false);
  shape.corners[1] = shadow_tile_corner_ortho(tilemap, tile_start + ivec2(extent.x, 0), false);
  shape.corners[2] = shadow_tile_corner_ortho(tilemap, tile_start + extent, false);
  shape.corners[3] = shadow_tile_corner_ortho(tilemap, tile_start + ivec2(0, extent.y), false);
  shape.corners[4] = shadow_tile_corner_ortho(tilemap, tile_start + ivec2(0, 0), true);
  shape.corners[5] = shadow_tile_corner_ortho(tilemap, tile_start + ivec2(extent.x, 0), true);
  shape.corners[6] = shadow_tile_corner_ortho(tilemap, tile_start + extent, true);
  shape.corners[7] = shadow_tile_corner_ortho(tilemap, tile_start + ivec2(0, extent.y), true);
  return shape;
}

/** \} */
