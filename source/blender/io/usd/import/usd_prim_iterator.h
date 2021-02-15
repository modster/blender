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

#include "usd_importer_context.h"

#include <pxr/usd/usd/common.h>

#include <map>
#include <set>
#include <vector>

struct Main;

namespace blender::io::usd {

class USDDataCache;
class USDXformableReader;

class USDPrimIterator {
 protected:
  pxr::UsdStageRefPtr stage_;
  USDImporterContext context_;
  Main *bmain_;

 public:
  USDPrimIterator(pxr::UsdStageRefPtr stage, const USDImporterContext &context, Main *bmain);

  USDPrimIterator(const char *file_path, const USDImporterContext &context, Main *bmain);

  bool valid() const;

  pxr::UsdStageRefPtr stage() const
  {
    return stage_;
  }

  void create_object_readers(std::vector<USDXformableReader *> &r_readers) const;

  void create_prototype_object_readers(
      std::map<pxr::SdfPath, USDXformableReader *> &r_proto_readers) const;

  void cache_prototype_data(USDDataCache &r_cache) const;

  void create_prototype_object_readers(
      std::map<pxr::SdfPath, std::vector<USDXformableReader *>> &r_proto_readers) const;

  bool gather_objects_paths(ListBase *object_paths) const;

  void debug_traverse_stage() const;

  USDXformableReader *get_object_reader(const pxr::UsdPrim &prim);

  static USDXformableReader *get_object_reader(const pxr::UsdPrim &prim,
                                               const USDImporterContext &context);

  static bool filter_by_purpose(const pxr::UsdPrim &prim, const USDImporterContext &context);

  static void create_object_readers(const pxr::UsdPrim &root,
                                    const USDImporterContext &context,
                                    std::vector<USDXformableReader *> &r_readers,
                                    std::vector<USDXformableReader *> &r_child_readers);

  static void debug_traverse_stage(const pxr::UsdStageRefPtr &usd_stage);
};

} /* namespace blender::io::usd */
