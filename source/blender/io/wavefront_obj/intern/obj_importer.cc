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

/** \file
 * \ingroup obj
 */

#include <string>

#include "BLI_float2.hh"
#include "BLI_float3.hh"
#include "BLI_map.hh"
#include "BLI_set.hh"
#include "BLI_string_ref.hh"

#include "obj_import_file_reader.hh"
#include "obj_import_mesh.hh"
#include "obj_import_nurbs.hh"
#include "obj_import_objects.hh"
#include "obj_importer.hh"

namespace blender::io::obj {

/**
 * Make Blender Mesh, Curve etc from Geometry and add them to the import collection.
 */
static void geometry_to_blender_objects(
    Main *bmain,
    Scene *scene,
    const OBJImportParams &import_params,
    Vector<std::unique_ptr<Geometry>> &all_geometries,
    const GlobalVertices &global_vertices,
    const Map<std::string, std::unique_ptr<MTLMaterial>> &materials)
{
  OBJImportCollection import_collection{bmain, scene};
  for (const std::unique_ptr<Geometry> &geometry : all_geometries) {
    if (geometry->get_geom_type() == GEOM_MESH) {
      MeshFromGeometry mesh_ob_from_geometry{*geometry, global_vertices};
      mesh_ob_from_geometry.create_mesh(bmain, materials, import_params);
      import_collection.add_object_to_collection(mesh_ob_from_geometry.mover());
    }
    else if (geometry->get_geom_type() == GEOM_CURVE) {
      CurveFromGeometry curve_ob_from_geometry(*geometry, global_vertices);
      curve_ob_from_geometry.create_curve(bmain, import_params);
      import_collection.add_object_to_collection(curve_ob_from_geometry.mover());
    }
  }
}

void importer_main(bContext *C, const OBJImportParams &import_params)
{
  Main *bmain = CTX_data_main(C);
  Scene *scene = CTX_data_scene(C);
  /* List of Geometry instances to be parsed from OBJ file. */
  Vector<std::unique_ptr<Geometry>> all_geometries;
  /* Container for vertex and UV vertex coordinates. */
  GlobalVertices global_vertices;
  /* List of MTLMaterial instances to be parsed from MTL file. */
  Map<std::string, std::unique_ptr<MTLMaterial>> materials;

  OBJParser obj_parser{import_params};
  obj_parser.parse_and_store(all_geometries, global_vertices);

  for (StringRef mtl_library : obj_parser.mtl_libraries()) {
    MTLParser mtl_parser{mtl_library, import_params.filepath};
    mtl_parser.parse_and_store(materials);
  }

  geometry_to_blender_objects(
      bmain, scene, import_params, all_geometries, global_vertices, materials);
}
}  // namespace blender::io::obj
