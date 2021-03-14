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

/** \file
 * \ingroup obj
 */

#pragma once

#include <array>

#include "BLI_float3.hh"
#include "BLI_map.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "DNA_node_types.h"

#include "MEM_guardedalloc.h"

#include "obj_export_mtl.hh"

namespace blender::io::obj {

struct UniqueNodeDeleter {
  void operator()(bNode *node)
  {
    MEM_freeN(node);
  }
};

struct UniqueNodetreeDeleter {
  void operator()(bNodeTree *node)
  {
    MEM_freeN(node);
  }
};

using unique_node_ptr = std::unique_ptr<bNode, UniqueNodeDeleter>;
using unique_nodetree_ptr = std::unique_ptr<bNodeTree, UniqueNodetreeDeleter>;

class ShaderNodetreeWrap {
 private:
  /* Node arrangement:
   * Texture Coordinates -> Mapping -> Image Texture -> (optional) Normal Map -> p-BSDF -> Material
   * Output. */
  unique_nodetree_ptr nodetree_;
  unique_node_ptr bsdf_;
  unique_node_ptr shader_output_;
  const MTLMaterial &mtl_mat_;

  /* List of all locations occupied by nodes. */
  Vector<std::array<int, 2>> node_locations;
  const float node_size_{300.f};

 public:
  ShaderNodetreeWrap(Main *bmain, const MTLMaterial &mtl_mat);
  ~ShaderNodetreeWrap();

  bNodeTree *get_nodetree();

 private:
  bNode *add_node_to_tree(const int node_type);
  std::pair<float, float> set_node_locations(const int pos_x);
  void link_sockets(unique_node_ptr from_node,
                    StringRef from_node_id,
                    bNode *to_node,
                    StringRef to_node_id,
                    const int from_node_pos_x);
  void set_bsdf_socket_values();
  void add_image_textures(Main *bmain);
};

constexpr eMTLSyntaxElement mtl_line_key_str_to_enum(const std::string_view key_str)
{
  if (key_str == "map_Kd") {
    return eMTLSyntaxElement::map_Kd;
  }
  if (key_str == "map_Ks") {
    return eMTLSyntaxElement::map_Ks;
  }
  if (key_str == "map_Ns") {
    return eMTLSyntaxElement::map_Ns;
  }
  if (key_str == "map_d") {
    return eMTLSyntaxElement::map_d;
  }
  if (key_str == "refl" || key_str == "map_refl") {
    return eMTLSyntaxElement::map_refl;
  }
  if (key_str == "map_Ke") {
    return eMTLSyntaxElement::map_Ke;
  }
  if (key_str == "map_Bump" || key_str == "bump") {
    return eMTLSyntaxElement::map_Bump;
  }
  return eMTLSyntaxElement::string;
}
}  // namespace blender::io::obj