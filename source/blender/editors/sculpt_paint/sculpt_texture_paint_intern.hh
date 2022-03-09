namespace blender::ed::sculpt_paint::texture_paint {

struct PixelData {
  struct {
    bool dirty : 1;
  } flags;
  int2 pixel_pos;
  float3 local_pos;
  float4 content;
};

struct NodeData {
  struct {
    bool dirty : 1;
  } flags;

  Vector<PixelData> pixels;
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
    for (PixelData &pixel : pixels) {
      if (pixel.flags.dirty) {
        const int pixel_offset = (pixel.pixel_pos[1] * image_buffer.x + pixel.pixel_pos[0]);
        copy_v4_v4(&image_buffer.rect_float[pixel_offset * 4], pixel.content);
        pixel.flags.dirty = false;
      }
    }
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
