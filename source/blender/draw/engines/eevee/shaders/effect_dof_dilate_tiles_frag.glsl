
/**
 * Tile dilate pass: Takes the 8x8 Tiles buffer and converts dilates the tiles with large CoC to
 * their neighboorhod. This pass is repeated multiple time until the maximum CoC can be covered.
 **/

#pragma BLENDER_REQUIRE(effect_dof_lib.glsl)

/* 1/16th of fullres. */
uniform sampler2D cocTilesFgBuffer;
uniform sampler2D cocTilesBgBuffer;

/* 1/16th of fullres. Same format as input. */
layout(location = 0) out vec2 outFgCoc; /* Min, Max */
layout(location = 1) out vec3 outBgCoc; /* Min, Max, MinIntersectable */

const float tile_to_fullres_factor = 16.0;

/* TODO should be a define. */
const float gather_ring_count = 3;
/* Error introduced by the random offset of the gathering kernel's center. */
const float bluring_radius_error = 1.0 + 1.0 / (gather_ring_count + 0.5);
/* NOTE(fclem): if we ever need larger gather. */
const int ring_width_multiplier = 1;

#define RINGS_COUNT 3

void main()
{
  ivec2 center_tile_pos = ivec2(gl_FragCoord.xy);

  CocTile ring_buckets[RINGS_COUNT];

  for (int ring = 0; ring < RINGS_COUNT; ring++) {
    ring_buckets[ring] = dof_coc_tile_init();

    int ring_distance = ring + 1;
    for (int sample_id = 0; sample_id < 4 * ring_distance; sample_id++) {
      ivec2 offset = dof_square_ring_sample_offset(ring_distance, sample_id);

      offset *= ring_width_multiplier;

      for (int i = 0; i < 2; i++) {
        ivec2 adj_tile_pos = center_tile_pos + ((i == 0) ? offset : -offset);

        CocTile adj_tile = dof_coc_tile_load(cocTilesFgBuffer, cocTilesBgBuffer, adj_tile_pos);

#ifdef DILATE_MODE_MIN_MAX
        /* Actually gather the "absolute" biggest coc but keeping the sign. */
        ring_buckets[ring].fg_min_coc = min(ring_buckets[ring].fg_min_coc, adj_tile.fg_min_coc);
        ring_buckets[ring].bg_max_coc = max(ring_buckets[ring].bg_max_coc, adj_tile.bg_max_coc);

#else /* DILATE_MODE_MIN_ABS */
        ring_buckets[ring].fg_max_coc = max(ring_buckets[ring].fg_max_coc, adj_tile.fg_max_coc);
        ring_buckets[ring].bg_min_coc = min(ring_buckets[ring].bg_min_coc, adj_tile.bg_min_coc);

        /* Should be tight as possible to reduce gather overhead (see slide 61). */
        float closest_neighbor_distance = length(max(abs(vec2(offset)) - 1.0, 0.0)) *
                                          tile_to_fullres_factor;

        ring_buckets[ring].bg_min_intersectable_coc = min(
            ring_buckets[ring].bg_min_intersectable_coc,
            adj_tile.bg_min_intersectable_coc + closest_neighbor_distance);
#endif
      }
    }
  }

  /* Load center tile. */
  CocTile out_tile = dof_coc_tile_load(cocTilesFgBuffer, cocTilesBgBuffer, center_tile_pos);

  for (int ring = 0; ring < RINGS_COUNT; ring++) {
    float ring_distance = float(ring + 1);

    ring_distance = (ring_distance * ring_width_multiplier - 1) * tile_to_fullres_factor;

    /* NOTE(fclem): Unsure if both sides of the inequalities have the same unit. */
#ifdef DILATE_MODE_MIN_MAX
    if (-ring_buckets[ring].fg_min_coc * bluring_radius_error > ring_distance) {
      out_tile.fg_min_coc = min(out_tile.fg_min_coc, ring_buckets[ring].fg_min_coc);
    }

    if (ring_buckets[ring].bg_max_coc * bluring_radius_error > ring_distance) {
      out_tile.bg_max_coc = max(out_tile.bg_max_coc, ring_buckets[ring].bg_max_coc);
    }

#else /* DILATE_MODE_MIN_ABS */
    /* Find minimum absolute CoC radii that will be intersected for the previously
     * computed maximum CoC values. */
    if (-out_tile.fg_min_coc * bluring_radius_error > ring_distance) {
      out_tile.fg_max_coc = max(out_tile.fg_max_coc, ring_buckets[ring].fg_max_coc);
    }

    if (out_tile.bg_max_coc * bluring_radius_error > ring_distance) {
      out_tile.bg_min_coc = min(out_tile.bg_min_coc, ring_buckets[ring].bg_min_coc);
      out_tile.bg_min_intersectable_coc = min(out_tile.bg_min_intersectable_coc,
                                              ring_buckets[ring].bg_min_intersectable_coc);
    }
#endif
  }

  outFgCoc = vec2(out_tile.fg_min_coc, out_tile.fg_max_coc);
  outBgCoc = vec3(out_tile.bg_min_coc, out_tile.bg_max_coc, out_tile.bg_min_intersectable_coc);
}
