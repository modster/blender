/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "BLI_math.h"
#include "BLI_math_vec_types.hh"
#include "BLI_rect.h"
#include "BLI_vector.hh"

#include "DNA_image_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_image.h"
#include "BKE_image_wrappers.hh"

#include "IMB_imbuf_types.h"

namespace blender::bke::pbvh::pixels {

/**
 * Loop indices. Only stores 2 indices, the third one is always `loop_indices[1] + 1`.
 */
struct EncodedLoopIndices {
  int2 encoded;

  EncodedLoopIndices(const int3 decoded) : encoded(decoded.x, decoded.y)
  {
  }

  int3 decode() const
  {
    return int3(encoded.x, encoded.y, encoded.y + 1);
  }
};

struct TrianglePaintInput {
  int3 vert_indices;
  /**
   * Delta barycentric coordinates between 2 neighbouring UV's in the U direction.
   *
   * Only the first two coordinates are stored. The third should be recalculated
   */
  float3 delta_barycentric_coord_u;
  /** Delta barycentric coordinates between 2 neighbouring UV's in the V direction. */
  float3 delta_barycentric_coord_v;

  /**
   * Initially only the vert indices are known.
   *
   * delta_barycentric_coord_u/v are initialized in a later stage as it requires image tile
   * dimensions.
   */
  TrianglePaintInput(const int3 vert_indices)
      : vert_indices(vert_indices),
        delta_barycentric_coord_u(0.0f, 0.0f, 0.0f),
        delta_barycentric_coord_v(0.0f, 0.0f, 0.0f)
  {
  }
};

/**
 * Data shared between pixels that belong to the same triangle.
 *
 * Data is stored as a list of structs, grouped by usage to improve performance (improves CPU
 * cache prefetching).
 */
struct Triangles {
  /** Data accessed by the inner loop of the painting brush. */
  Vector<TrianglePaintInput> paint_input;
  /** Per triangle the index of the polygon it belongs to. */
  Vector<int> poly_indices;
  /**
   * Loop indices per triangle.
   *
   * NOTE: is only available during building the triangles. Kept here as in the future we need
   * the data to calculate normals.
   */
  Vector<EncodedLoopIndices> loop_indices;

 public:
  void append(const int3 vert_indices, const EncodedLoopIndices loop_indices, const int poly_index)
  {
    this->paint_input.append(TrianglePaintInput(vert_indices));
    this->loop_indices.append(loop_indices);
    this->poly_indices.append(poly_index);
  }

  int3 get_loop_indices(const int index) const
  {
    return loop_indices[index].decode();
  }

  int get_poly_index(const int index)
  {
    return poly_indices[index];
  }

  TrianglePaintInput &get_paint_input(const int index)
  {
    return paint_input[index];
  }

  const TrianglePaintInput &get_paint_input(const int index) const
  {
    return paint_input[index];
  }

  void clear()
  {
    paint_input.clear();
    loop_indices.clear();
    poly_indices.clear();
  }

  uint64_t size() const
  {
    return paint_input.size();
  }

  uint64_t mem_size() const
  {
    return loop_indices.size() * sizeof(EncodedLoopIndices) +
           paint_input.size() * sizeof(TrianglePaintInput) + poly_indices.size() * sizeof(int);
  }
};

/**
 * Encode sequential pixels to reduce memory footprint.
 */
struct PackedPixelRow {
  /** Barycentric coordinate of the first pixel. */
  float3 start_barycentric_coord;
  /** Image coordinate starting of the first pixel. */
  ushort2 start_image_coordinate;
  /** Number of sequential pixels encoded in this package. */
  ushort num_pixels;
  /** Reference to the pbvh triangle index. */
  ushort triangle_index;
};

struct TileData {
  short tile_number;
  struct {
    bool dirty : 1;
  } flags;

  /* Dirty region of the tile in image space. */
  rcti dirty_region;

  Vector<PackedPixelRow> pixel_rows;

  TileData()
  {
    flags.dirty = false;
    BLI_rcti_init_minmax(&dirty_region);
  }

  void mark_region(Image &image, const image::ImageTileWrapper &image_tile, ImBuf &image_buffer)
  {
    BKE_image_partial_update_mark_region(
        &image, image_tile.image_tile, &image_buffer, &dirty_region);
    BLI_rcti_init_minmax(&dirty_region);
    flags.dirty = false;
  }
};

struct NodeData {
  struct {
    bool dirty : 1;
  } flags;

  rctf uv_region;

  Vector<TileData> tiles;
  Triangles triangles;

  NodeData()
  {
    flags.dirty = false;
  }

  void init_pixels_rasterization(Object *ob, PBVHNode *node, ImBuf *image_buffer);

  TileData *find_tile_data(const image::ImageTileWrapper &image_tile)
  {
    for (TileData &tile : tiles) {
      if (tile.tile_number == image_tile.get_tile_number()) {
        return &tile;
      }
    }
    return nullptr;
  }

  void mark_region(Image &image, const image::ImageTileWrapper &image_tile, ImBuf &image_buffer)
  {
    TileData *tile = find_tile_data(image_tile);
    if (tile) {
      tile->mark_region(image, image_tile, image_buffer);
    }
  }

  void clear_data()
  {
    tiles.clear();
    triangles.clear();
  }

  static void free_func(void *instance)
  {
    NodeData *node_data = static_cast<NodeData *>(instance);
    MEM_delete(node_data);
  }
};

Triangles &BKE_pbvh_pixels_triangles_get(PBVHNode &node);
TileData *BKE_pbvh_pixels_tile_data_get(PBVHNode &node, const image::ImageTileWrapper &image_tile);
void BKE_pbvh_pixels_mark_dirty(PBVHNode &node);
void BKE_pbvh_pixels_mark_image_dirty(PBVHNode &node, Image &image, ImageUser &image_user);
/** Extend pixels to fix uv seams for the given nodes. */
void BKE_pbvh_pixels_fix_seams(PBVH &pbvh,
                               Image &image,
                               ImageUser &image_user,
                               const MLoopUV *ldata_uv);

}  // namespace blender::bke::pbvh::pixels
