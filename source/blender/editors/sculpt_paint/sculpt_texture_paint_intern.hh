namespace blender::ed::sculpt_paint::texture_paint {

struct PixelData {
  int2 pixel_pos;
  float3 local_pos;
  float4 content;
};

struct Pixels {
  Vector<int2> image_coordinates;
  Vector<float3> local_positions;
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
    printf("%s", __func__);
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
};

}  // namespace blender::ed::sculpt_paint::texture_paint
