/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2020 Blender Foundation.
 * All rights reserved.
 */
#pragma once

#include "usd.h"

#include <pxr/usd/usdShade/material.h>

struct Main;
struct Material;
struct bNode;
struct bNodeTree;

namespace blender::io::usd {

// Helper struct used when arranging nodes in columns, keeping track the
// occupancy information for a given column.  I.e., for column n,
// column_offsets[n] is the y-offset (from top to bottom) of the occupied
// region in that column.
struct NodePlacementContext {
  float origx;
  float origy;
  std::vector<float> column_offsets;
  const float horizontal_step;

  NodePlacementContext(float in_origx, float in_origy, float in_horizontal_step = 300.0f)
      : origx(in_origx),
        origy(in_origy),
        column_offsets(64, 0.0f),
        horizontal_step(in_horizontal_step)
  {
  }
};

/* Converts USD materials to Blender representation. */

// The current implementation converts UsdPreviewSurface to Blender
// nodes as follows:
//
// UsdPreviewSurface -> Pricipled BSDF
// UsdUVTexture -> Texture Image + Normal Map
// UsdPrimvarReader_float2 -> UV Map
//
// Limitations: arbitrary primvar readers or UsdTransform2d not yet
// supported. For UsdPreviewSurface, only the file and st inputs
// are handled, and the color space is retrieved from the texture
// metadata.
//
// TODO(makowalski):  Investigate adding support for converting additional
// shaders and inputs.  Supporting certain types of inputs, such as texture
// scale and bias, will probably require creating Blender Group nodes with
// the corresponding inputs.

class USDMaterialReader {
 protected:
  USDImportParams params_;

  Main *bmain_;

 public:
  USDMaterialReader(const USDImportParams &params, Main *bmain);

  Material *add_material(const pxr::UsdShadeMaterial &usd_material) const;

  void import_usd_preview(Material *mtl, const pxr::UsdShadeMaterial &usd_material) const;

  void import_usd_preview(Material *mtl, const pxr::UsdShadeShader &usd_material) const;

  void set_node_input(const pxr::UsdShadeInput &usd_input,
                      bNode *dest_node,
                      const char *dest_socket_name,
                      bNodeTree *ntree,
                      int column,
                      NodePlacementContext &r_ctx) const;

  void convert_usd_uv_texture(const pxr::UsdShadeShader &usd_shader,
                              const pxr::TfToken &usd_source_name,
                              bNode *dest_node,
                              const char *dest_socket_name,
                              bNodeTree *ntree,
                              int column,
                              NodePlacementContext &r_ctx) const;

  void convert_usd_primvar_reader_float2(const pxr::UsdShadeShader &usd_shader,
                                         const pxr::TfToken &usd_source_name,
                                         bNode *dest_node,
                                         const char *dest_socket_name,
                                         bNodeTree *ntree,
                                         int column,
                                         NodePlacementContext &r_ctx) const;
};

}  // namespace blender::io::usd
