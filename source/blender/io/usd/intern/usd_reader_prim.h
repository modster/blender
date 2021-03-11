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
 */

/** \file
 * \ingroup busd
 */

#ifndef __USD_READER_OBJECT_H__
#define __USD_READER_OBJECT_H__

#include "usd.h"

#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdUtils/sparseValueWriter.h>

#include <vector>

extern "C" {
#include "DNA_ID.h"
}

struct CacheFile;
struct Main;
struct Mesh;
struct Object;

struct ImportSettings {
  bool do_convert_mat;
  float conversion_mat[4][4];

  int from_up;
  int from_forward;
  float scale;
  bool is_sequence;
  bool set_frame_range;

  /* Length and frame offset of file sequences. */
  int sequence_len;
  int sequence_offset;

  /* From MeshSeqCacheModifierData.read_flag */
  int read_flag;

  bool validate_meshes;

  CacheFile *cache_file;

  float vel_scale;

  ImportSettings()
      : do_convert_mat(false),
        from_up(0),
        from_forward(0),
        scale(1.0f),
        is_sequence(false),
        set_frame_range(false),
        sequence_len(1),
        sequence_offset(0),
        read_flag(0),
        validate_meshes(false),
        cache_file(NULL),
        vel_scale(1.0f)
  {
  }
};

// Most generic USD Reader

class USDPrimReader {

 protected:
  std::string m_name;
  std::string m_prim_path;
  Object *m_object;
  pxr::UsdPrim m_prim;
  pxr::UsdStageRefPtr m_stage;
  const USDImportParams &m_import_params;
  USDPrimReader *m_parent_reader;

  ImportSettings *m_settings;

  int m_refcount;

 public:
  USDPrimReader(pxr::UsdStageRefPtr stage,
                const pxr::UsdPrim &object,
                const USDImportParams &import_params,
                ImportSettings &settings);
  virtual ~USDPrimReader();

  const pxr::UsdPrim &prim() const;

  virtual bool valid() const;

  virtual void createObject(Main *bmain, double motionSampleTime);
  virtual void readObjectData(Main *bmain, double motionSampleTime);

  Object *object() const;
  void object(Object *ob);

  USDPrimReader *parent() const
  {
    return m_parent_reader;
  }
  void parent(USDPrimReader *parent)
  {
    m_parent_reader = parent;
  }

  int refcount() const;
  void incref();
  void decref();

  virtual void addCacheModifier();

  const std::string &name() const
  {
    return m_name;
  }
  const std::string &prim_path() const
  {
    return m_prim_path;
  }
};

#endif /* __USD_READER_OBJECT_H__ */
