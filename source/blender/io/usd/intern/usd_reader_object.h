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
#pragma once

#include "usd.h"
#include "usd_importer_context.h"

#include <pxr/usd/usd/prim.h>

#include <string>
#include <vector>

struct Main;
struct Mesh;
struct Object;

namespace blender::io::usd {

class UsdObjectReader {
protected:
  std::string m_name;
  std::string m_object_name;
  std::string m_data_name;
  Object *m_object;
  pxr::UsdPrim m_prim;

  USDImporterContext m_context;

  double m_min_time;
  double m_max_time;

  /* Use reference counting since the same reader may be used by multiple
    * modifiers and/or constraints. */
  int m_refcount;

public:
  typedef std::vector<UsdObjectReader *> ptr_vector;

  explicit UsdObjectReader(const pxr::UsdPrim &prim, const USDImporterContext &context);

  virtual ~UsdObjectReader();

  const pxr::UsdPrim &prim() const;

  Object *object() const;

  void setObject(Object *ob);

  const std::string &name() const
  {
    return m_name;
  }
  const std::string &object_name() const
  {
    return m_object_name;
  }
  const std::string &data_name() const
  {
    return m_data_name;
  }

  virtual bool valid() const = 0;

  virtual void readObjectData(Main *bmain, double time) = 0;

  virtual struct Mesh *read_mesh(struct Mesh *mesh,
                                 double time,
                                 int read_flag,
                                 const char **err_str);

  virtual bool topology_changed(Mesh *existing_mesh,
                                double time);

  /** Reads the object matrix and sets up an object transform if animated. */
  void setupObjectTransform(const double time);

  double minTime() const;
  double maxTime() const;

  int refcount() const;
  void incref();
  void decref();

  void read_matrix(float r_mat[4][4], const double time, const float scale, bool &is_constant);

};

}  // namespace blender::io::alembic
