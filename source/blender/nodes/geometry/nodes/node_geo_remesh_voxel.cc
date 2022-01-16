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

#include "UI_interface.h"

#include "BKE_mesh_remesh_voxel.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_remesh_voxel_cc {
static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>(N_("Mesh")).supported_type(GEO_COMPONENT_TYPE_MESH);
  b.add_input<decl::Float>(N_("Voxel Size")).default_value(1.0f).min(0.01f).max(FLT_MAX);
  b.add_input<decl::Float>(N_("Adaptivity")).default_value(0.0f).min(0.0f).max(1.0f);
  b.add_output<decl::Geometry>(N_("Mesh"));
}

static void node_geo_exec(GeoNodeExecParams params)
{
#ifdef WITH_OPENVDB
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh");
  const float voxel_size = params.extract_input<float>("Voxel Size");
  const float adaptivity = params.extract_input<float>("Adaptivity");
  geometry_set.modify_geometry_sets([&](GeometrySet &geometry_set) {
    const Mesh *mesh_in = geometry_set.get_mesh_for_read();
    if (mesh_in != nullptr) {
      Mesh *mesh_out = BKE_mesh_remesh_voxel(mesh_in, voxel_size, adaptivity, 0.0f);
      geometry_set.replace_mesh(mesh_out);
    }
  });

  params.set_output("Mesh", std::move(geometry_set));
#else
  params.error_message_add(NodeWarningType::Error,
                           TIP_("Disabled, Blender was compiled without OpenVDB"));
  params.set_default_remaining_outputs();
#endif
}
}  // namespace blender::nodes::node_geo_remesh_voxel_cc

void register_node_type_geo_remesh_voxel()
{
  namespace file_ns = blender::nodes::node_geo_remesh_voxel_cc;

  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_REMESH_VOXEL, "Voxel Remesh", NODE_CLASS_GEOMETRY);
  ntype.declare = file_ns::node_declare;
  ntype.geometry_node_execute = file_ns::node_geo_exec;
  nodeRegisterType(&ntype);
}
