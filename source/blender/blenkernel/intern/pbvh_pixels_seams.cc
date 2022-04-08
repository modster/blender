/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BKE_pbvh.h"
#include "BKE_pbvh.hh"

#include "pbvh_intern.h"

#include "BLI_math_geom.h"

namespace blender::bke::pbvh::pixels {
using namespace blender::bke::image;

bool intersect_uv_pixel(const ushort2 &pixel_coordinate,
                        const ImBuf &image_buffer,
                        const float2 triangle_uvs[3])
{

  float2 uv_bottom_left = float2(pixel_coordinate.x / float(image_buffer.x),
                                 pixel_coordinate.y / float(image_buffer.y));
  float2 uv_top_right = float2((pixel_coordinate.x + 1) / float(image_buffer.x),
                               (pixel_coordinate.y + 1) / float(image_buffer.y));
  float2 uv_bottom_right(uv_top_right[0], uv_bottom_left[1]);
  float2 uv_top_left(uv_bottom_left[0], uv_top_right[1]);

  return (isect_seg_seg_v2_simple(
              uv_bottom_left, uv_bottom_right, triangle_uvs[0], triangle_uvs[1]) ||
          isect_seg_seg_v2_simple(uv_bottom_left, uv_top_left, triangle_uvs[0], triangle_uvs[1]) ||
          isect_seg_seg_v2_simple(uv_top_left, uv_top_right, triangle_uvs[0], triangle_uvs[1]) ||
          isect_seg_seg_v2_simple(
              uv_bottom_right, uv_top_right, triangle_uvs[0], triangle_uvs[1])) ||
         (isect_seg_seg_v2_simple(
              uv_bottom_left, uv_bottom_right, triangle_uvs[1], triangle_uvs[2]) ||
          isect_seg_seg_v2_simple(uv_bottom_left, uv_top_left, triangle_uvs[1], triangle_uvs[2]) ||
          isect_seg_seg_v2_simple(uv_top_left, uv_top_right, triangle_uvs[1], triangle_uvs[2]) ||
          isect_seg_seg_v2_simple(
              uv_bottom_right, uv_top_right, triangle_uvs[1], triangle_uvs[2])) ||
         (isect_seg_seg_v2_simple(
              uv_bottom_left, uv_bottom_right, triangle_uvs[2], triangle_uvs[0]) ||
          isect_seg_seg_v2_simple(uv_bottom_left, uv_top_left, triangle_uvs[2], triangle_uvs[0]) ||
          isect_seg_seg_v2_simple(uv_top_left, uv_top_right, triangle_uvs[2], triangle_uvs[0]) ||
          isect_seg_seg_v2_simple(
              uv_bottom_right, uv_top_right, triangle_uvs[2], triangle_uvs[0]));
}

struct UVSeamExtenderRowPackage {
  /** Amount of pixels to extend beyond the determined extension to reduce rendering artifacts. */
  static const int ADDITIONAL_EXTEND_X = 1;

  PixelsPackage *package;
  TrianglePaintInput *triangle_paint_data;
  bool is_new;
  int extend_xmin_len = 0;
  int extend_xmax_len = 0;

  UVSeamExtenderRowPackage(PixelsPackage *package,
                           TrianglePaintInput *triangle_paint_data,
                           const ImBuf &image_buffer,
                           bool is_new,
                           const int3 loop_indices,
                           const MLoopUV *ldata_uv)
      : package(package), triangle_paint_data(triangle_paint_data), is_new(is_new)
  {
    init_extend_x_len(image_buffer, loop_indices, ldata_uv);
  }

  bool should_extend_start() const
  {
    return extend_xmin_len != 0;
  }

  void extend_x_start()
  {
    BLI_assert(extend_xmin_len != 0);
    package->num_pixels += 1;
    package->start_image_coordinate[0] -= 1;
    package->start_barycentric_coord -= float2(triangle_paint_data->add_barycentric_coord_x);
    extend_xmin_len--;
  }

  bool should_extend_end() const
  {
    return extend_xmax_len != 0;
  }

  void extend_x_end()
  {
    BLI_assert(extend_xmax_len != 0);
    package->num_pixels += 1;
    extend_xmax_len--;
  }

 private:
  void init_extend_x_len(const ImBuf &image_buffer,
                         const int3 loop_indices,
                         const MLoopUV *ldata_uv)
  {
    if (!is_new) {
      return;
    }

    float2 triangle_uvs[3];
    triangle_uvs[0] = ldata_uv[loop_indices[0]].uv;
    triangle_uvs[1] = ldata_uv[loop_indices[1]].uv;
    triangle_uvs[2] = ldata_uv[loop_indices[2]].uv;

    init_extend_xmin_len(image_buffer, triangle_uvs);
    init_extend_xmax_len(image_buffer, triangle_uvs);
    BLI_assert(extend_xmin_len);
    BLI_assert(extend_xmax_len);
  }

  void init_extend_xmin_len(const ImBuf &image_buffer, const float2 triangle_uvs[3])
  {
    int result = 0;
    while (should_extend_xmin(image_buffer, result, triangle_uvs)) {
      result++;
    }
    extend_xmin_len = result + ADDITIONAL_EXTEND_X;
  }

  bool should_extend_xmin(const ImBuf &image_buffer,
                          int offset,
                          const float2 triangle_uvs[3]) const
  {
    uint16_t pixel_offset = offset + 1;
    ushort2 pixel = package->start_image_coordinate - ushort2(pixel_offset, 0);
    return intersect_uv_pixel(pixel, image_buffer, triangle_uvs);
  }

  void init_extend_xmax_len(const ImBuf &image_buffer, const float2 triangle_uvs[3])
  {
    int result = 0;
    while (should_extend_xmax(image_buffer, result, triangle_uvs)) {
      result++;
    }
    extend_xmax_len = result + ADDITIONAL_EXTEND_X;
  }

  bool should_extend_xmax(const ImBuf &image_buffer,
                          int offset,
                          const float2 triangle_uvs[3]) const
  {
    uint16_t pixel_offset = offset + 1;
    ushort2 pixel = package->start_image_coordinate + ushort2(pixel_offset, 0);
    return intersect_uv_pixel(pixel, image_buffer, triangle_uvs);
  }
};

class UVSeamExtenderRow : public Vector<UVSeamExtenderRowPackage> {

 public:
  bool has_packages_that_needs_fixing = false;

  void append(UVSeamExtenderRowPackage &package)
  {
    has_packages_that_needs_fixing |= package.is_new;
    Vector<UVSeamExtenderRowPackage>::append(package);
  }

  void extend_x(int image_buffer_width)
  {
    std::sort(
        begin(), end(), [](const UVSeamExtenderRowPackage &a, const UVSeamExtenderRowPackage &b) {
          return a.package->start_image_coordinate[0] < b.package->start_image_coordinate[0];
        });
    extend_x_start();
    extend_x_end(image_buffer_width);
  }

 private:
  void extend_x_start()
  {
    int prev_package_x = -1;
    int index = 0;

    for (UVSeamExtenderRowPackage &package : *this) {
      if (package.is_new) {
        while (package.should_extend_start()) {
          if (package.package->start_image_coordinate[0] - 1 <= prev_package_x) {
            /* No room left for extending. */
            break;
          }
          package.extend_x_start();
        }
      }

      prev_package_x = package.package->start_image_coordinate[0] + package.package->num_pixels;
      index++;
    }
  }

  void extend_x_end(int image_buffer_width)
  {
    int index = 0;
    for (UVSeamExtenderRowPackage &package : *this) {
      if (package.is_new) {
        int next_package_x;
        if (index < size() - 1) {
          const UVSeamExtenderRowPackage &next_package = (*this)[index + 1];
          next_package_x = next_package.package->start_image_coordinate[0];
        }
        else {
          next_package_x = image_buffer_width + 1;
        }
        while (package.should_extend_end()) {
          if (package.package->start_image_coordinate[0] + package.package->num_pixels >=
              next_package_x - 1) {
            /* No room left for extending */
            break;
          }
          package.extend_x_end();
        }
      }

      index++;
    }
  }
};

class UVSeamExtender {
  Vector<UVSeamExtenderRow> rows;
  int image_buffer_width_;

 public:
  explicit UVSeamExtender(PBVH &pbvh,
                          const ImageTileWrapper &image_tile,
                          const ImBuf &image_buffer,
                          const MLoopUV *ldata_uv)
      : image_buffer_width_(image_buffer.x)
  {
    rows.resize(image_buffer.y);
    init(pbvh, image_tile, image_buffer, ldata_uv);
  }

  void extend_x()
  {
    for (UVSeamExtenderRow &row : rows) {
      if (row.has_packages_that_needs_fixing) {
        row.extend_x(image_buffer_width_);
      }
    }
  }

 private:
  void init(PBVH &pbvh,
            const ImageTileWrapper &image_tile,
            const ImBuf &image_buffer,
            const MLoopUV *ldata_uv)
  {
    for (int n = 0; n < pbvh.totnode; n++) {
      PBVHNode &node = pbvh.nodes[n];
      if ((node.flag & PBVH_Leaf) == 0) {
        continue;
      }
      init(node, image_tile, image_buffer, ldata_uv);
    }
  }

  void init(PBVHNode &node,
            const ImageTileWrapper &image_tile,
            const ImBuf &image_buffer,
            const MLoopUV *ldata_uv)
  {
    NodeData &node_data = *static_cast<NodeData *>(node.pixels.node_data);
    TileData *tile_node_data = node_data.find_tile_data(image_tile);
    if (tile_node_data == nullptr) {
      return;
    }
    init(node, node_data, *tile_node_data, image_buffer, ldata_uv);
  }

  void init(PBVHNode &node,
            NodeData &node_data,
            TileData &tile_data,
            const ImBuf &image_buffer,
            const MLoopUV *ldata_uv)
  {
    for (PixelsPackage &package : tile_data.packages) {
      UVSeamExtenderRowPackage row_package(
          &package,
          &node_data.triangles.get_paint_input(package.triangle_index),
          image_buffer,
          (node.flag & PBVH_UpdatePixels) != 0,
          node_data.triangles.get_loop_indices(package.triangle_index),
          ldata_uv);
      append(row_package);
    }
  }

  void append(UVSeamExtenderRowPackage &package)
  {
    rows[package.package->start_image_coordinate[1]].append(package);
  }
};

/** Extend pixels to fix uv seams for the given nodes. */
void BKE_pbvh_pixels_fix_seams(PBVH &pbvh,
                               Image &image,
                               ImageUser &image_user,
                               const MLoopUV *ldata_uv)
{
  ImageUser local_image_user = image_user;
  LISTBASE_FOREACH (ImageTile *, tile_data, &image.tiles) {
    image::ImageTileWrapper image_tile(tile_data);
    local_image_user.tile = image_tile.get_tile_number();
    ImBuf *image_buffer = BKE_image_acquire_ibuf(&image, &local_image_user, NULL);
    if (image_buffer == nullptr) {
      continue;
    }
    UVSeamExtender extender(pbvh, image_tile, *image_buffer, ldata_uv);
    extender.extend_x();
    BKE_image_release_ibuf(&image, image_buffer, NULL);
  }
}

}  // namespace blender::bke::pbvh::pixels
