/* SPDX-License-Identifier: GPL-2.0-or-later */

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

/* TODO(jbakker): move encoders to blenlib. */
class EncodedBarycentricWeights;

/** Barycentric weights. */
class BarycentricWeights {
 private:
  float3 weights;

 public:
  explicit BarycentricWeights(const float2 v1, const float2 v2, const float2 v3, const float2 co)
  {
    barycentric_weights_v2(v1, v2, v3, co, weights);
  }

  explicit BarycentricWeights(const float3 weights) : weights(weights)
  {
  }

  const bool is_inside_triangle() const
  {
    return barycentric_inside_triangle_v2(weights);
  }

  float3 operator-(const BarycentricWeights &rhs) const
  {
    return weights - rhs.weights;
  }

  BarycentricWeights operator+(const float3 &rhs) const
  {
    return BarycentricWeights(weights + rhs);
  }

  BarycentricWeights &operator-=(const float3 &rhs)
  {
    weights -= rhs;
    return *this;
  }

  BarycentricWeights &operator+=(const float3 &rhs)
  {
    weights += rhs;
    return *this;
  }

  operator const float *() const
  {
    return weights;
  }

  friend class EncodedBarycentricWeights;
};

/**
 * Barycentric weights encoded into 2 floats.
 *
 * The third coordinate can be extracted as all 3 coordinates should sum up to 1.
 *
 * \code{.cc}
 * co[2] = 1.0 - co[0] - co[1]
 * \endcode
 */
class EncodedBarycentricWeights {
 private:
  float2 encoded;

 public:
  EncodedBarycentricWeights &operator=(const BarycentricWeights decoded)
  {
    encoded = float2(decoded.weights.x, decoded.weights.y);
    return *this;
  }

  const BarycentricWeights decode() const
  {
    return BarycentricWeights(float3(encoded.x, encoded.y, 1.0 - encoded.x - encoded.y));
  }

  EncodedBarycentricWeights &operator-=(const float2 &other)
  {
    encoded -= other;
    return *this;
  }
};

/**
 * Loop incides. Only stores 2 indices, the third one is always `loop_indices[1] + 1`.
 * Second could be delta encoded with the first loop index.
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

struct Triangle {
  int3 loop_indices;
  int3 vert_indices;
  int poly_index;
  float3 add_barycentric_coord_x;
};

struct TrianglePaintInput {
  int3 vert_indices;
  float3 add_barycentric_coord_x;
  float3 add_barycentric_coord_y;

  TrianglePaintInput(const Triangle &triangle)
      : vert_indices(triangle.vert_indices),
        add_barycentric_coord_x(triangle.add_barycentric_coord_x)
  {
  }
};

/**
 * Pixels of the same triangle share some data.
 *
 * Data is stored as a list of structs, grouped by usage to improve performance as it improves CPU
 * cache prefetching.
 *
 */
struct Triangles {
  /** Paint input per triangle. */
  Vector<TrianglePaintInput> paint_input;
  /** Polygon index per triangle. */
  Vector<int> poly_indices;
  /**
   * Loop indices per triangle.
   * NOTE: is only available during building the triangles.
   */
  Vector<EncodedLoopIndices> loop_indices;

 public:
  void append(const Triangle &triangle)
  {
    paint_input.append(TrianglePaintInput(triangle));
    loop_indices.append(triangle.loop_indices);
    poly_indices.append(triangle.poly_index);
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
 * Encode multiple sequential pixels to reduce memory footprint.
 */
struct PixelsPackage {
  /** Barycentric coordinate of the first encoded pixel. */
  EncodedBarycentricWeights start_barycentric_coord;
  /** Image coordinate starting of the first encoded pixel. */
  ushort2 start_image_coordinate;
  /** Number of sequetial pixels encoded in this package. */
  ushort num_pixels;
  /** Reference to the pbvh triangle index. */
  ushort triangle_index;
};

class PixelPackages : public Vector<PixelsPackage> {
 public:
  /* Sort the packages to improve CPU cache utilization during painting. */
  void sort();
};

struct TileData {
  short tile_number;
  struct {
    bool dirty : 1;
  } flags;

  /* Dirty region of the tile in image space. */
  rcti dirty_region;

  PixelPackages packages;

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
