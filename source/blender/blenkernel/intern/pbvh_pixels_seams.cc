/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BKE_pbvh.h"
#include "BKE_pbvh.hh"

#include "pbvh_intern.h"

namespace blender::bke::pbvh::pixels {
using namespace blender::bke::image;

struct UVSeamExtenderRowPackage {

  PixelsPackage *package;
  TrianglePaintInput *triangle_paint_data;
  bool is_new;

  UVSeamExtenderRowPackage(PixelsPackage *package,
                           TrianglePaintInput *triangle_paint_data,
                           bool is_new)
      : package(package), triangle_paint_data(triangle_paint_data), is_new(is_new)
  {
  }

  void extend_x_start()
  {
    package->num_pixels += 1;
    package->start_image_coordinate[0] -= 1;
    package->start_barycentric_coord -= float2(triangle_paint_data->add_barycentric_coord_x);
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

  void extend_x()
  {
    std::sort(
        begin(), end(), [](const UVSeamExtenderRowPackage &a, const UVSeamExtenderRowPackage &b) {
          return a.package->start_image_coordinate[0] < b.package->start_image_coordinate[0];
        });

    int prev_package_x = 0;
    int index = 0;
    for (UVSeamExtenderRowPackage &package : *this) {
      if (package.is_new) {
        if (package.package->start_image_coordinate[0] - 1 > prev_package_x) {
          package.extend_x_start();
        }
      }

      prev_package_x = package.package->start_image_coordinate[0] + package.package->num_pixels;
      index++;
    }
  }
};

class UVSeamExtender {
  Vector<UVSeamExtenderRow> rows;

 public:
  explicit UVSeamExtender(PBVH &pbvh, ImageTileWrapper &image_tile, ImBuf &image_buffer)
  {
    rows.resize(image_buffer.y);
    init(pbvh, image_tile);
  }

  void extend_x()
  {
    for (UVSeamExtenderRow &row : rows) {
      if (row.has_packages_that_needs_fixing) {
        row.extend_x();
      }
    }
  }

 private:
  void init(PBVH &pbvh, ImageTileWrapper &image_tile)
  {
    for (int n = 0; n < pbvh.totnode; n++) {
      PBVHNode &node = pbvh.nodes[n];
      if ((node.flag & PBVH_Leaf) == 0) {
        continue;
      }
      init(node, image_tile);
    }
  }

  void init(PBVHNode &node, ImageTileWrapper &image_tile)
  {
    NodeData &node_data = *static_cast<NodeData *>(node.pixels.node_data);
    TileData *tile_node_data = node_data.find_tile_data(image_tile);
    if (tile_node_data == nullptr) {
      return;
    }
    init(node, node_data, *tile_node_data);
  }

  void init(PBVHNode &node, NodeData &node_data, TileData &tile_data)
  {
    for (PixelsPackage &package : tile_data.packages) {
      UVSeamExtenderRowPackage row_package(
          &package,
          &node_data.triangles.get_paint_input(package.triangle_index),
          (node.flag & PBVH_Leaf) == 0);
      append(row_package);
    }
  }

  void append(UVSeamExtenderRowPackage &package)
  {
    rows[package.package->start_image_coordinate[1]].append(package);
  }
};

/** Extend pixels to fix uv seams for the given nodes. */
void BKE_pbvh_pixels_fix_seams(PBVH &pbvh, Image &image, ImageUser &image_user)
{
  ImageUser local_image_user = image_user;
  LISTBASE_FOREACH (ImageTile *, tile_data, &image.tiles) {
    image::ImageTileWrapper image_tile(tile_data);
    local_image_user.tile = image_tile.get_tile_number();
    ImBuf *image_buffer = BKE_image_acquire_ibuf(&image, &local_image_user, NULL);
    if (image_buffer == nullptr) {
      continue;
    }
    UVSeamExtender extender(pbvh, image_tile, *image_buffer);
    extender.extend_x();
    BKE_image_release_ibuf(&image, image_buffer, NULL);
  }
}

}  // namespace blender::bke::pbvh::pixels
