namespace blender::ed::sculpt_paint::texture_paint {

struct Polygon {
};

struct Triangle {
  int3 loop_indices;
  int3 vert_indices;
  int poly_index;
  float3 add_barycentric_coord_x;
  float automasking_factor;
};

/**
 * Encode multiple sequential pixels to reduce memory footprint.
 * 
 * Memory footprint can be reduced more.
 * - Only store 2 barycentric coordinates. the third one can be extracted
 *   from the 2 known ones.
 * - start_image_coordinate can be a delta encoded with the previous package.
 *   initial coordinate could be at the triangle level or pbvh.
 * - num_pixels could be delta encoded or at least be a short.
 * - triangle index could be delta encoded.
 * - encode everything in using variable bits per structure.
 *   first byte would indicate the number of bytes used per element.
 */
struct PixelsPackage {
  /** Image coordinate starting of the first encoded pixel. */
  int2 start_image_coordinate;
  /** Barycentric coordinate of the first encoded pixel. */
  float3 start_barycentric_coord;
  /** Number of sequetial pixels encoded in this package. */
  int num_pixels;
  /** Reference to the pbvh triangle index. */
  int triangle_index;
};

struct Pixel {
  float3 pos;
  float2 uv;

  Pixel &operator+=(const Pixel &other)
  {
    pos += other.pos;
    uv += other.uv;
    return *this;
  }

  Pixel operator-(const Pixel &other)
  {
    Pixel result;
    result.pos = pos - other.pos;
    result.uv = uv - other.uv;
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

struct NodeData {
  struct {
    bool dirty : 1;
  } flags;

  // Vector<PixelData> pixels;
  Pixels pixels;
  rcti dirty_region;
  rctf uv_region;

  Vector<Triangle> triangles;
  Vector<PixelsPackage> encoded_pixels;

  NodeData()
  {
    flags.dirty = false;
    BLI_rcti_init_minmax(&dirty_region);
  }

  void init_pixels_rasterization(Object *ob, PBVHNode *node, ImBuf *image_buffer);

  void flush(ImBuf &image_buffer)
  {
    flags.dirty = false;
    for (int i = 0; i < pixels.size(); i++) {

      if (pixels.is_dirty(i)) {
        const int2 &image_coord = pixels.image_coord(i);
        const int pixel_offset = (image_coord[1] * image_buffer.x + image_coord[0]);
        const float4 &color = pixels.color(i);
        copy_v4_v4(&image_buffer.rect_float[pixel_offset * 4], color);
      }
    }
    pixels.clear_dirty();
  }

  void mark_region(Image &image, ImBuf &image_buffer)
  {
    print_rcti_id(&dirty_region);
    BKE_image_partial_update_mark_region(
        &image, static_cast<ImageTile *>(image.tiles.first), &image_buffer, &dirty_region);
    BLI_rcti_init_minmax(&dirty_region);
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
