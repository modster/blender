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

#include "BKE_mesh.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_modifier_types.h"

#include "GEO_mesh_remesh_blocks.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_remesh_cc {
static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>(N_("Mesh")).supported_type(GEO_COMPONENT_TYPE_MESH);
  b.add_input<decl::Int>(N_("Depth")).default_value(4).min(2).max(64);
  b.add_input<decl::Float>(N_("Scale")).default_value(0.9f).min(0.0f).max(0.99f);
  b.add_input<decl::Float>(N_("Threshold")).default_value(1.0f).min(0.01f).max(FLT_MAX);
  b.add_output<decl::Geometry>(N_("Mesh"));
}

static void node_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "remesh_blocks_mode", 0, "", ICON_NONE);
}

static void geo_remesh_blocks_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  node->custom1 = 0;
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh");
  const char flag = 0;
  const char mode = params.node().custom1;
  const int hermite_num = 1;
  const int depth = params.extract_input<int>("Depth");
  const float scale = min_ff(params.extract_input<float>("Scale"), 0.99f);
  const float threshold = params.extract_input<float>("Threshold");

  if (geometry_set.has_mesh()) {
    const Mesh *input_mesh = geometry_set.get_mesh_for_read();

    Mesh *output_mesh = GEO_mesh_remesh_blocks(
        input_mesh, flag, mode, threshold, hermite_num, scale, depth);

    BKE_mesh_copy_parameters_for_eval(output_mesh, input_mesh);
    BKE_mesh_calc_edges(output_mesh, true, false);
    BKE_mesh_normals_tag_dirty(output_mesh);

    geometry_set.replace_mesh(output_mesh);
  }
  params.set_output("Mesh", std::move(geometry_set));
}
}  // namespace blender::nodes::node_geo_remesh_cc

void register_node_type_geo_remesh_blocks()
{

  namespace file_ns = blender::nodes::node_geo_remesh_cc;

  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_REMESH_BLOCKS, "Remesh Blocks", NODE_CLASS_GEOMETRY);
  ntype.declare = file_ns::node_declare;
  node_type_init(&ntype, file_ns::geo_remesh_blocks_init);
  ntype.geometry_node_execute = file_ns::node_geo_exec;
  ntype.draw_buttons = file_ns::node_layout;
  nodeRegisterType(&ntype);
}
