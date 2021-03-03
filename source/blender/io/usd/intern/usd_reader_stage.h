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

/** \file
 * \ingroup busd
 */

#ifndef __USD_READER_ARCHIVE_H__
#define __USD_READER_ARCHIVE_H__

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

/* Wrappers around input and output archives. The goal is to be able to use
 * streams so that unicode paths work on Windows (T49112), and to make sure that
 * the stream objects remain valid as long as the archives are open.
 */

class USDStageReader {
 public:
  typedef std::map<pxr::SdfPath, std::vector<USDPrimReader *>> ProtoReaderMap;

 protected:
  pxr::UsdStageRefPtr m_stage;
  USDImportParams m_params;
  ImportSettings m_settings;

  std::vector<USDPrimReader *> m_readers;

  // Readers for scenegraph instance prototypes.
  ProtoReaderMap m_proto_readers;

 public:
  USDStageReader(struct Main *bmain, const char *filename);
  ~USDStageReader();

  std::vector<USDPrimReader *> collect_readers(struct Main *bmain,
                                               const USDImportParams &params,
                                               ImportSettings &settings);

  bool valid() const;

  pxr::UsdStageRefPtr stage()
  {
    return m_stage;
  }
  USDImportParams &params()
  {
    return m_params;
  }
  ImportSettings &settings()
  {
    return m_settings;
  }

  void params(USDImportParams &a_params)
  {
    m_params = a_params;
  }
  void settings(ImportSettings &a_settings)
  {
    m_settings = a_settings;
  }

  void clear_readers();

  void clear_proto_readers(bool decref);

  const ProtoReaderMap &proto_readers() const
  {
    return m_proto_readers;
  };
};

};  // namespace blender::io::usd

#endif /* __USD_READER_ARCHIVE_H__ */
