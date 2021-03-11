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

#include "usd_reader_volume.h"

extern "C" {
#include "DNA_cachefile_types.h"
#include "DNA_camera_types.h"
#include "DNA_constraint_types.h"
#include "DNA_modifier_types.h"
#include "DNA_object_force_types.h"
#include "DNA_object_types.h"
#include "DNA_space_types.h" /* for FILE_MAX */
#include "DNA_volume_types.h"

#include "BKE_constraint.h"
#include "BKE_lib_id.h"
#include "BKE_library.h"
#include "BKE_material.h"
#include "BKE_mesh.h"
#include "BKE_modifier.h"
#include "BKE_node.h"
#include "BKE_object.h"
#include "BKE_volume.h"

#include "BLI_listbase.h"
#include "BLI_math.h"
#include "BLI_math_geom.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "RNA_access.h"

#include "WM_api.h"
#include "WM_types.h"
}

#include <pxr/pxr.h>
#include <pxr/usd/usdVol/openVDBAsset.h>
#include <pxr/usd/usdVol/volume.h>

#include <iostream>

namespace blender::io::usd {

void USDVolumeReader::create_object(Main *bmain, double motionSampleTime)
{
  Volume *volume = (Volume *)BKE_volume_add(bmain, name_.c_str());
  id_us_min(&volume->id);

  object_ = BKE_object_add_only_object(bmain, OB_VOLUME, name_.c_str());
  object_->data = volume;
}

void USDVolumeReader::read_object_data(Main *bmain, double motionSampleTime)
{
  volume_ = pxr::UsdVolVolume(prim_);

  if (!volume_) {
    return;
  }

  pxr::UsdVolVolume::FieldMap fields = volume_.GetFieldPaths();

  std::string filepath;

  Volume *volume = (Volume *)object_->data;
  VolumeGrid *defaultGrid = BKE_volume_grid_active_get(volume);

  for (auto it = fields.begin(); it != fields.end(); ++it) {

    pxr::UsdPrim fieldPrim = prim_.GetStage()->GetPrimAtPath(it->second);

    if (fieldPrim.IsA<pxr::UsdVolOpenVDBAsset>()) {
      pxr::UsdVolOpenVDBAsset fieldBase(fieldPrim);

      pxr::UsdAttribute filepathAttr = fieldBase.GetFilePathAttr();
      pxr::UsdAttribute fieldNameAttr = fieldBase.GetFieldNameAttr();

      std::string fieldName = "density";

      if (fieldNameAttr.IsAuthored()) {
        fieldNameAttr.Get(&fieldName, 0.0);

        // A Blender volume creates density by default
        if (fieldName != "density") {
          defaultGrid = BKE_volume_grid_add(volume, fieldName.c_str(), VOLUME_GRID_FLOAT);
        }
      }

      if (filepathAttr.IsAuthored()) {

        pxr::SdfAssetPath fp;
        filepathAttr.Get(&fp, 0.0);

        if (filepathAttr.ValueMightBeTimeVarying()) {
          std::vector<double> filePathTimes;
          filepathAttr.GetTimeSamples(&filePathTimes);

          int start = (int)filePathTimes.front();
          int end = (int)filePathTimes.back();

          volume->is_sequence = (char)true;
          volume->frame_start = start;
          volume->frame_duration = (end - start) + 1;
        }

        filepath = fp.GetResolvedPath();

        strcpy(volume->filepath, filepath.c_str());
      }
    }
  }

  USDXformReader::read_object_data(bmain, motionSampleTime);
}

}  // namespace blender::io::usd
