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
 * along with this program; if not, write to the Free Software  Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2016 Kévin Dietrich.
 * All rights reserved.
 */
#pragma once

struct Main;
struct Scene;

#include "usd.h"
#include "usd_reader_prim.h"

#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>

#include <map>
#include <vector>

struct ImportSettings;

namespace blender::io::usd {

typedef std::map<pxr::SdfPath, std::vector<USDPrimReader *>> ProtoReaderMap;

class USDStageReader {

 protected:
  pxr::UsdStageRefPtr stage_;
  USDImportParams params_;
  ImportSettings settings_;

  std::vector<USDPrimReader *> readers_;

  // Readers for scenegraph instance prototypes.
  ProtoReaderMap proto_readers_;

 public:
  USDStageReader(struct Main *bmain, const char *filename);
  ~USDStageReader();

  static USDPrimReader *create_reader(const pxr::UsdPrim &prim,
                                      const USDImportParams &params,
                                      ImportSettings &settings);

  // This version of create_reader() does not filter by primitive type.  I.e.,
  // it will convert any prim to a reader, if possible, regardless of the
  // primitive types specified by the user in the import options.
  static USDPrimReader *create_reader(class USDStageReader *archive, const pxr::UsdPrim &prim);

  void collect_readers(struct Main *bmain,
                       const USDImportParams &params,
                       ImportSettings &settings);

  bool valid() const;

  pxr::UsdStageRefPtr stage()
  {
    return stage_;
  }
  USDImportParams &params()
  {
    return params_;
  }
  ImportSettings &settings()
  {
    return settings_;
  }

  void params(USDImportParams &a_params)
  {
    params_ = a_params;
  }
  void settings(ImportSettings &a_settings)
  {
    settings_ = a_settings;
  }

  void clear_readers(bool decref = true);

  void clear_proto_readers(bool decref = true);

  const ProtoReaderMap &proto_readers() const
  {
    return proto_readers_;
  };

  const std::vector<USDPrimReader *> &readers() const
  {
    return readers_;
  };
};

};  // namespace blender::io::usd
