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

#include "usd_reader_object.h"
#include "usd_util.h"

#include "DNA_cachefile_types.h"
#include "DNA_constraint_types.h"
#include "DNA_modifier_types.h"
#include "DNA_space_types.h" /* for FILE_MAX */

#include "BKE_constraint.h"
#include "BKE_lib_id.h"
#include "BKE_modifier.h"
#include "BKE_object.h"

#include "BLI_listbase.h"
#include "BLI_math_geom.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/usd/usdGeom/xformable.h>

#include <iostream>


namespace blender::io::usd {

UsdObjectReader::UsdObjectReader(const pxr::UsdPrim &prim, const USDImporterContext &context)
  : m_name(""),
  m_object_name(""),
  m_data_name(""),
  m_object(NULL),
  m_prim(prim),
  m_context(context),
  m_min_time(std::numeric_limits<double>::max()),
  m_max_time(std::numeric_limits<double>::min()),
  m_refcount(0)
{
  m_name = prim.GetPath().GetString();

  std::vector<std::string> parts;
  split(m_name, '/', parts);

  if (parts.size() >= 2)
  {
    m_object_name = parts[parts.size() - 2];
    m_data_name = parts[parts.size() - 1];
  }
  else if (!parts.empty())
  {
    m_object_name = m_data_name = parts[parts.size() - 1];
  }
}


UsdObjectReader::~UsdObjectReader()
{
}

const pxr::UsdPrim &UsdObjectReader::prim() const
{
  return m_prim;
}

Object *UsdObjectReader::object() const
{
  return m_object;
}

void UsdObjectReader::setObject(Object *ob)
{
  m_object = ob;
}

struct Mesh *UsdObjectReader::read_mesh(struct Mesh *existing_mesh,
                                        double UNUSED(time),
                                        int UNUSED(read_flag),
                                        const char **UNUSED(err_str))
{
  return existing_mesh;
}

bool UsdObjectReader::topology_changed(Mesh * /*existing_mesh*/,
                                        double /*time*/)
{
  /* The default implementation of read_mesh() just returns the original mesh, so never changes the
    * topology. */
  return false;
}

void UsdObjectReader::setupObjectTransform(const double time)
{
  if (!this->m_object)
  {
    return;
  }

  bool is_constant = false;
  float transform_from_usd[4][4];

  this->read_matrix(transform_from_usd, time, this->m_context.import_params.scale, is_constant);

  /* Apply the matrix to the object. */
  BKE_object_apply_mat4(m_object, transform_from_usd, true, false);
  BKE_object_to_mat4(m_object, m_object->obmat);

  // TODO:  Set up transform constraint if not constant.
}

void UsdObjectReader::read_matrix(float r_mat[4][4] /* local matrix */,
                                  const double time,
                                  const float scale,
                                  bool &is_constant)
{
  pxr::UsdGeomXformable xformable(m_prim);

  if (!xformable) {
    unit_m4(r_mat);
    is_constant = true;
    return;
  }

  // TODO:  Check for constant transform.

  pxr::GfMatrix4d usd_local_to_world = xformable.ComputeLocalToWorldTransform(time);

  double double_mat[4][4];

  usd_local_to_world.Get(double_mat);

  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      r_mat[i][j] = static_cast<float>(double_mat[i][j]);
    }
  }

  if (this->m_context.stage_up_axis == pxr::UsdGeomTokens->y)
  {
    // Swap the matrix from y-up to z-up.
    copy_m44_axis_swap(r_mat, r_mat, USD_ZUP_FROM_YUP);
  }

  float scale_mat[4][4];
  scale_m4_fl(scale_mat, scale);
  mul_m4_m4m4(r_mat, scale_mat, r_mat);
}

double UsdObjectReader::minTime() const
{
  return m_min_time;
}

double UsdObjectReader::maxTime() const
{
  return m_max_time;
}

int UsdObjectReader::refcount() const
{
  return m_refcount;
}

void UsdObjectReader::incref()
{
  m_refcount++;
}

void UsdObjectReader::decref()
{
  m_refcount--;
  BLI_assert(m_refcount >= 0);
}

}  // namespace blender::io::usd
