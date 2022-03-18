#include "IMB_imbuf_wrappers.hh"

namespace blender::ed::sculpt_paint::texture_paint {

struct Polygon {
};

#if 0
/* Stores a barycentric coordinate in a float2. */
struct EncodedBarycentricCoord {
  float2 encoded;

  EncodedBarycentricCoord &operator=(const float3 decoded)
  {
    encoded = float2(decoded.x, decoded.y);
    return *this;
  }

  float3 decode() const
  {
    return float3(encoded.x, encoded.y, 1.0 - encoded.x - encoded.y);
  }
};
#else
struct EncodedBarycentricCoord {
  ushort2 encoded;

  EncodedBarycentricCoord &operator=(const float3 decoded)
  {
    encoded = ushort2(encode(decoded.x), encode(decoded.y));
    return *this;
  }

  float3 decode() const
  {
    float2 decoded(decode(encoded.x), decode(encoded.y));
    return float3(decoded.x, decoded.y, 1.0f - decoded.x - decoded.y);
  }

  static uint16_t encode(float value)
  {
    float clamped = clamp_f(value, 0.0f, 1.0f);
    return (uint16_t)clamped * 65535.0;
  }

  static float decode(uint16_t value)
  {
    return value / 65535.0f;
  }
};
#endif

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
  float automasking_factor;
};

struct Triangles {
  Vector<EncodedLoopIndices> loop_indices;
  Vector<int3> vert_indices;
  Vector<int> poly_indices;
  Vector<float3> add_barycentric_coords_x;
  Vector<float> automasking_factors;

 public:
  void append(Triangle &triangle)
  {
    loop_indices.append(triangle.loop_indices);
    vert_indices.append(triangle.vert_indices);
    poly_indices.append(triangle.poly_index);
    add_barycentric_coords_x.append(float3(0.0f));
    automasking_factors.append(0.0);
  }

  int3 get_vert_indices(const int index) const
  {
    return vert_indices[index];
  }
  int3 get_loop_indices(const int index) const
  {
    return loop_indices[index].decode();
  }
  int get_poly_index(const int index)
  {
    return poly_indices[index];
  }

  void set_add_barycentric_coord_x(const int index, const float3 add_barycentric_coord_x)
  {
    add_barycentric_coords_x[index] = add_barycentric_coord_x;
  }
  float3 get_add_barycentric_coord_x(const int index) const
  {
    return add_barycentric_coords_x[index];
  }

  void set_automasking_factor(const int index, const float automasking_factor)
  {
    automasking_factors[index] = automasking_factor;
  }
  float get_automasking_factor(const int index) const
  {
    return automasking_factors[index];
  }

  void cleanup_after_init()
  {
    loop_indices.clear();
  }

  uint64_t size()
  {
    return vert_indices.size();
  }
  uint64_t mem_size()
  {
    return loop_indices.size() * sizeof(EncodedLoopIndices) + vert_indices.size() * sizeof(int3) +
           poly_indices.size() * sizeof(int) + add_barycentric_coords_x.size() * sizeof(float3) +
           automasking_factors.size() * sizeof(float);
  }
};

/**
 * Encode multiple sequential pixels to reduce memory footprint.
 *
 * Memory footprint can be reduced more.
 * v Only store 2 barycentric coordinates. the third one can be extracted
 *   from the 2 known ones.
 * - start_image_coordinate can be a delta encoded with the previous package.
 *   initial coordinate could be at the triangle level or pbvh.
 * v num_pixels could be delta encoded or at least be a short.
 * X triangle index could be delta encoded.
 * - encode everything in using variable bits per structure.
 *   first byte would indicate the number of bytes used per element.
 * - only store triangle index when it changes.
 */
struct PixelsPackage {
  /** Barycentric coordinate of the first encoded pixel. */
  EncodedBarycentricCoord start_barycentric_coord;
  /** Image coordinate starting of the first encoded pixel. */
  ushort2 start_image_coordinate;
  /** Number of sequetial pixels encoded in this package. */
  ushort num_pixels;
  /** Reference to the pbvh triangle index. */
  ushort triangle_index;
};

struct Pixel {
  /** object local position of the pixel on the surface. */
  float3 pos;

  Pixel &operator+=(const Pixel &other)
  {
    pos += other.pos;
    return *this;
  }

  Pixel operator-(const Pixel &other) const
  {
    Pixel result;
    result.pos = pos - other.pos;
    return result;
  }
};

struct PixelData {
  int2 pixel_pos;
  float3 local_pos;
  float3 weights;
  int3 vertices;
  float4 content;
};

struct Pixels {
  Vector<int2> image_coordinates;
  Vector<float3> local_positions;
  Vector<int3> vertices;
  Vector<float3> weights;

  Vector<float4> colors;
  std::vector<bool> dirty;

  uint64_t size() const
  {
    return image_coordinates.size();
  }

  bool is_dirty(uint64_t index) const
  {
    return dirty[index];
  }

  const int2 &image_coord(uint64_t index) const
  {
    return image_coordinates[index];
  }
  const float3 &local_position(uint64_t index) const
  {
    return local_positions[index];
  }

  const float3 local_position(uint64_t index, MVert *mvert) const
  {
    const int3 &verts = vertices[index];
    const float3 &weight = weights[index];
    const float3 &pos1 = mvert[verts.x].co;
    const float3 &pos2 = mvert[verts.y].co;
    const float3 &pos3 = mvert[verts.z].co;
    float3 local_pos;
    interp_v3_v3v3v3(local_pos, pos1, pos2, pos3, weight);
    return local_pos;
  }

  const float4 &color(uint64_t index) const
  {
    return colors[index];
  }
  float4 &color(uint64_t index)
  {
    return colors[index];
  }

  void clear_dirty()
  {
    dirty = std::vector<bool>(size(), false);
  }

  void mark_dirty(uint64_t index)
  {
    dirty[index] = true;
  }

  void append(PixelData &pixel)
  {
    image_coordinates.append(pixel.pixel_pos);
    local_positions.append(pixel.local_pos);
    colors.append(pixel.content);
    weights.append(pixel.weights);
    vertices.append(pixel.vertices);
    dirty.push_back(false);
  }
};
struct TileData {
  short tile_number;
  struct {
    bool dirty : 1;
  } flags;

  /* Dirty region of the tile in image space. */
  rcti dirty_region;

  Vector<PixelsPackage> encoded_pixels;

  TileData()
  {
    flags.dirty = false;
    BLI_rcti_init_minmax(&dirty_region);
  }

  void mark_region(Image &image, const imbuf::ImageTileWrapper &image_tile, ImBuf &image_buffer)
  {
    print_rcti_id(&dirty_region);
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

  TileData *find_tile_data(const imbuf::ImageTileWrapper &image_tile)
  {
    for (TileData &tile : tiles) {
      if (tile.tile_number == image_tile.get_tile_number()) {
        return &tile;
      }
    }
    return nullptr;
  }

  void mark_region(Image &image, const imbuf::ImageTileWrapper &image_tile, ImBuf &image_buffer)
  {
    TileData *tile = find_tile_data(image_tile);
    if (tile) {
      tile->mark_region(image, image_tile, image_buffer);
    }
  }

  static void free_func(void *instance)
  {
    NodeData *node_data = static_cast<NodeData *>(instance);
    MEM_delete(node_data);
  }
};

struct TexturePaintingUserData {
  Object *ob;
  Brush *brush;
  PBVHNode **nodes;
  std::vector<bool> vertex_brush_tests;
  Vector<float> automask_factors;
};

}  // namespace blender::ed::sculpt_paint::texture_paint
