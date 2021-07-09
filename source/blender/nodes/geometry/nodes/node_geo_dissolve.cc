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

#include "UI_interface.h"
#include "UI_resources.h"

#include "bmesh.h"
#include "bmesh_tools.h"
#include "math.h"
#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_dissolve_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Angle"), M_PI * 0.25, 0.0f, 0.0f, 1.0f, 0.0f, M_PI, PROP_ANGLE},
    {SOCK_STRING, N_("Delimiter")},
    {SOCK_BOOLEAN, N_("All Boundaries"), false},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_dissolve_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_dissolve_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "delimiter", 0, nullptr, ICON_NONE);
}

static void geo_node_dissolve_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryDissolve *node_storage = (NodeGeometryDissolve *)MEM_callocN(
      sizeof(NodeGeometryDissolve), __func__);

  node->storage = node_storage;
  node_storage->delimiter = GEO_NODE_DISSOLVE_DELIMITTER_SELECTION_BORDER;
}

namespace blender::nodes {
static Mesh *dissolve_mesh(const float angle,
                           const bool all_boundaries,
                           const int delimiter_type,
                           const Array<bool> &delimiter,
                           Mesh *mesh)
{
  const BMeshCreateParams bmesh_create_params = {0};
  const BMeshFromMeshParams bmesh_from_mesh_params = {
      true, 0, 0, 0, {CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX, CD_MASK_ORIGINDEX}};
  BMesh *bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);
  if (delimiter_type & GEO_NODE_DISSOLVE_DELIMITTER_SELECTION_BORDER) {
    BM_temporary_tag_faces(bm, delimiter.data());
  }
  else {
    BM_temporary_tag_edges(bm, delimiter.data());
  }
  BM_mesh_decimate_dissolve(bm, angle, all_boundaries, (BMO_Delimit)delimiter_type);

  Mesh *result = BKE_mesh_from_bmesh_for_eval_nomain(bm, NULL, mesh);
  BM_mesh_free(bm);
  result->runtime.cd_dirty_vert |= CD_MASK_NORMAL;
  return result;
}

static void geo_node_dissolve_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  const float angle = params.extract_input<float>("Angle");

  if (geometry_set.has_mesh()) {
    MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();

    Mesh *input_mesh = mesh_component.get_for_write();

    if (input_mesh->totvert <= 3) {
      params.error_message_add(NodeWarningType::Error,
                               TIP_("Node requires mesh with more than 3 input faces"));
    }

    const bool all_boundaries = params.extract_input<bool>("All Boundaries");
    const bNode &node = params.node();
    const NodeGeometryDissolve &node_storage = *(NodeGeometryDissolve *)node.storage;

    const bool default_delimiter = false;
    AttributeDomain delimiter_domain = ATTR_DOMAIN_FACE;
    int delimiter_domain_size = input_mesh->totpoly;
    if (node_storage.delimiter & GEO_NODE_DISSOLVE_DELIMITTER_SELECTION) {
      delimiter_domain = ATTR_DOMAIN_EDGE;
      delimiter_domain_size = input_mesh->totedge;
    };

    const GVArrayPtr delimiter = params.get_input_attribute(
        "Delimiter", mesh_component, delimiter_domain, CD_PROP_BOOL, &default_delimiter);
    if (!delimiter) {
      return;
    }
    const GVArray_Typed<bool> delimiter_as_typed = delimiter->typed<bool>();
    Array<bool> mask(delimiter_domain_size);
    for (const int i : delimiter_as_typed.index_range()) {
      mask[i] = delimiter_as_typed[i];
    }

    Mesh *result = dissolve_mesh(angle, all_boundaries, node_storage.delimiter, mask, input_mesh);
    geometry_set.replace_mesh(result);
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_dissolve()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_DISSOLVE, "Dissolve", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_dissolve_in, geo_node_dissolve_out);
  node_type_storage(
      &ntype, "NodeGeometryDissolve", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, geo_node_dissolve_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_dissolve_exec;
  ntype.draw_buttons = geo_node_dissolve_layout;
  nodeRegisterType(&ntype);
}
