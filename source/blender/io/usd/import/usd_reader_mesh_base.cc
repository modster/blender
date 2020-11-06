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

#include "usd_reader_mesh_base.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "BKE_lib_id.h"
#include "BKE_main.h"
#include "BKE_mesh.h"
#include "BKE_object.h"

#include <pxr/usd/usdGeom/imageable.h>

#include <iostream>

namespace blender::io::usd {

std::map<pxr::SdfPath, Mesh *> USDMeshReaderBase::s_prototype_meshes;

USDMeshReaderBase::USDMeshReaderBase(const pxr::UsdPrim &prim, const USDImporterContext &context)
    : USDXformableReader(prim, context)
{
}

USDMeshReaderBase::~USDMeshReaderBase()
{
}

void USDMeshReaderBase::create_object(Main *bmain, double time)
{
  if (!this->valid()) {
    return;
  }

  /* Determine mesh visibility.
   * TODO(makowalski): Consider optimizations to avoid this expensive call,
   * for example, by determining visibility during stage traversal. */

  pxr::UsdGeomImageable imageable(this->prim_);

  if (!imageable) {
    std::cerr << "Warning:  Invalid mesh imageable schema for " << this->prim_path_ << std::endl;
    return;
  }

  pxr::TfToken vis_tok = imageable.ComputeVisibility();

  if (vis_tok == pxr::UsdGeomTokens->invisible) {
    return;
  }

  std::string obj_name = merged_with_parent_ ? this->parent_prim_name() : this->prim_name();

  if (obj_name.empty()) {
    /* Sanity check. */
    std::cerr << "Warning: couldn't determine object name for " << this->prim_path() << std::endl;
  }

  object_ = BKE_object_add_only_object(bmain, OB_MESH, obj_name.c_str());
  Mesh *mesh = this->read_mesh(bmain, time);
  object_->data = mesh;

  if (this->context_.import_params.import_materials) {
    assign_materials(bmain, mesh, time);
  }
}

Mesh *USDMeshReaderBase::read_mesh(Main *bmain, double time)
{
  /* If this prim is an instance proxy and instancing is enabled,
   * return the shared mesh created by the instance prototype. */

  if (this->context_.import_params.use_instancing && this->context_.proto_readers &&
      this->prim_.IsInstanceProxy()) {

    pxr::UsdPrim proto_prim = this->prim_.GetPrimInMaster();

    if (proto_prim) {

      pxr::SdfPath proto_path = proto_prim.GetPath();

      /* See if the prototype is already been cached. */
      std::map<pxr::SdfPath, Mesh *>::const_iterator proto_mesh_iter = s_prototype_meshes.find(
          proto_path);
      if (proto_mesh_iter != s_prototype_meshes.end()) {
        Mesh *cached_mesh = proto_mesh_iter->second;
        if (cached_mesh) {
          /* Increment the user count before returning. */
          id_us_plus(&cached_mesh->id);
        }
        return cached_mesh;
      }

      /* No cached mesh.  Lookup the reader for the prototype prim. */

      ObjectReaderMap::iterator proto_reader_iter = this->context_.proto_readers->find(proto_path);

      if (proto_reader_iter != this->context_.proto_readers->end()) {

        USDXformableReader *proto_reader = proto_reader_iter->second;

        USDMeshReaderBase *proto_mesh_reader = dynamic_cast<USDMeshReaderBase *>(proto_reader);

        if (proto_mesh_reader) {
          Mesh *proto_mesh = proto_mesh_reader->create_mesh(bmain, time);

          if (proto_mesh) {
            s_prototype_meshes.insert(std::make_pair(proto_path, proto_mesh));
            return proto_mesh;
          }
          else {
            std::cerr << "Couldn't evaluate prototype " << proto_path.GetString()
                      << " mesh for instance " << this->prim_path_ << std::endl;
          }
        }
        else {
          std::cerr << "Invalid prototype " << proto_path.GetString()
                    << " reader type for instance " << this->prim_path_ << std::endl;
        }
      }
    }
  }

  /* Not sharing the prototype mesh, so create unique mesh data. */
  return create_mesh(bmain, time);
}

}  // namespace blender::io::usd
