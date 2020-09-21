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

#include "BLI_float3.hh"
#include "BLI_map.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "DNA_node_types.h"

namespace blender::io::obj {
class OBJMesh;

/**
 * Generic container for texture node properties.
 */
struct tex_map_XX {
  tex_map_XX(StringRef to_socket_id) : dest_socket_id(to_socket_id){};

  /** Target socket which this texture node connects to. */
  const std::string dest_socket_id{};
  float3 translation{0.0f};
  float3 scale{1.0f};
  /* Only Flat and Smooth projections are supported. */
  int projection_type = SHD_PROJ_FLAT;
  std::string image_path{};
  std::string mtl_dir_path;
};

/**
 * Container suited for storing Material data for/from a MTL file.
 */
struct MTLMaterial {
  MTLMaterial()
  {
    texture_maps.add("map_Kd", tex_map_XX("Base Color"));
    texture_maps.add("map_Ks", tex_map_XX("Specular"));
    texture_maps.add("map_Ns", tex_map_XX("Roughness"));
    texture_maps.add("map_d", tex_map_XX("Alpha"));
    texture_maps.add("map_refl", tex_map_XX("Metallic"));
    texture_maps.add("map_Ke", tex_map_XX("Emission"));
    texture_maps.add("map_Bump", tex_map_XX("Normal"));
  }

  /**
   * Return a reference to the texture map corresponding to the given ID
   * Caller must ensure that the lookup key given exists in the Map.
   */
  tex_map_XX &tex_map_of_type(StringRef map_string)
  {
    {
      BLI_assert(texture_maps.contains_as(map_string));
      return texture_maps.lookup_as(map_string);
    }
  }

  std::string name{};
  /* Always check for negative values while importing or exporting. Use defaults if
   * any value is negative. */
  float Ns{-1.0f};
  float3 Ka{-1.0f};
  float3 Kd{-1.0f};
  float3 Ks{-1.0f};
  float3 Ke{-1.0f};
  float Ni{-1.0f};
  float d{-1.0f};
  int illum{-1};
  Map<const std::string, tex_map_XX> texture_maps;
  /** Only used for Normal Map node: map_Bump. */
  float map_Bump_strength{-1.0f};
};

/**
 * Get an Object's material properties from `Material` as well as `bNodeTree`.
 */
class MaterialWrap {
 private:
  /**
   * One of the object's materials, to be exported.
   */
  const Material *export_mtl_ = nullptr;
  /**
   * First Principled-BSDF node encountered in the object's node tree.
   */
  bNode *bsdf_node_ = nullptr;

 public:
  void fill_materials(const OBJMesh &obj_mesh_data, Vector<MTLMaterial> &r_mtl_materials);

 private:
  void init_bsdf_node(StringRefNull object_name);
  void store_bsdf_properties(MTLMaterial &r_mtl_mat) const;
  void store_image_textures(MTLMaterial &r_mtl_mat) const;
};
}  // namespace blender::io::obj
