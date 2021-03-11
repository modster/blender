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

#include "usd_reader_prim.h"

extern "C" {
#include "DNA_cachefile_types.h"
#include "DNA_constraint_types.h"
#include "DNA_modifier_types.h"
#include "DNA_space_types.h" /* for FILE_MAX */

#include "BKE_constraint.h"
#include "BKE_library.h"
#include "BKE_modifier.h"
#include "BKE_object.h"

#include "BLI_listbase.h"
#include "BLI_math_geom.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "WM_api.h"
#include "WM_types.h"
}

namespace blender::io::usd {

USDPrimReader::USDPrimReader(pxr::UsdStageRefPtr stage,
                             const pxr::UsdPrim &object,
                             const USDImportParams &import_params,
                             ImportSettings &settings)
    : name_(object.GetName().GetString()),
      prim_path_(object.GetPrimPath().GetString()),
      object_(nullptr),
      prim_(object),
      stage_(stage),
      import_params_(import_params),
      parent_reader_(nullptr),
      settings_(&settings),
      refcount_(0)
{
  //@TODO(bjs): This should be handled better
  if (name_ == "/")
    name_ = "root";
}

USDPrimReader::~USDPrimReader()
{
}

const pxr::UsdPrim &USDPrimReader::prim() const
{
  return prim_;
}

Object *USDPrimReader::object() const
{
  return object_;
}

void USDPrimReader::object(Object *ob)
{
  object_ = ob;
}

bool USDPrimReader::valid() const
{
  return prim_.IsValid();
}

int USDPrimReader::refcount() const
{
  return refcount_;
}

void USDPrimReader::incref()
{
  refcount_++;
}

void USDPrimReader::decref()
{
  refcount_--;
  BLI_assert(refcount_ >= 0);
}

}  // namespace blender::io::usd
