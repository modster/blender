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
#include "usd_importer_context.h"
#include "usd_reader_prim.h"

#include <string>
#include <vector>

struct CacheFile;
struct Main;
struct Mesh;
struct Object;

namespace blender::io::usd {

class USDDataCache;

/* Wraps the UsdGeomXformable schema. Abstract base class for all readers
 * that create a Blender object and compute its transform. */

class USDXformableReader : public USDPrimReader {

 protected:
  Object *object_;

  USDXformableReader *parent_;

  bool merged_with_parent_;

 public:
  explicit USDXformableReader(const pxr::UsdPrim &prim, const USDImporterContext &context);

  virtual ~USDXformableReader();

  Object *object() const;

  void set_object(Object *object)
  {
    object_ = object;
  }

  USDXformableReader *parent() const
  {
    return parent_;
  }

  void set_parent(USDXformableReader *par)
  {
    parent_ = par;
  }

  void set_merged_with_parent(bool flag)
  {
    merged_with_parent_ = flag;
  }

  bool merged_with_parent() const
  {
    return merged_with_parent_;
  }

  /* Determines whether or not this reader takes into
   * account the encapsulated USD primitive's parent when
   * computing the object's transform and sets the value
   * of the merged_with_parent_ flag accordingly. Note that
   * this function may be expensive to compute. */
  void eval_merged_with_parent();

  std::string get_object_name() const
  {
    return merged_with_parent_ ? this->parent_prim_name() : this->prim_name();
  }

  std::string get_data_name() const
  {
    return this->prim_name();
  }

  virtual bool valid() const = 0;

  virtual void create_object(Main *bmain, double time, USDDataCache *data_cache) = 0;

  virtual bool can_merge_with_parent() const;

  void set_object_transform(const double time, CacheFile *cache_file = nullptr);

  virtual void read_matrix(float r_mat[4][4],
                           const double time,
                           const float scale,
                           bool &is_constant) const;
};

} /* namespace blender::io::usd */
