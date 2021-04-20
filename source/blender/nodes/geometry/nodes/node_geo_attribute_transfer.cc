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
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_attribute_transfer_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_GEOMETRY, N_("Target")},
    {SOCK_STRING, N_("Source")},
    {SOCK_STRING, N_("Destination")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_transfer_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_attribute_transfer_layout(uiLayout *layout,
                                               bContext *UNUSED(C),
                                               PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "domain", 0, IFACE_("Domain"), ICON_NONE);
  uiItemR(layout, ptr, "mapping", 0, IFACE_("Mapping"), ICON_NONE);
}

namespace blender::nodes {

static void geo_node_attribute_transfer_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryAttributeTransfer *data = (NodeGeometryAttributeTransfer *)MEM_callocN(
      sizeof(NodeGeometryAttributeTransfer), __func__);
  node->storage = data;
}

static void geo_node_attribute_transfer_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet target_geometry_set = params.extract_input<GeometrySet>("Target");
  const std::string src_attribute_name = params.extract_input<std::string>("Source");
  const std::string dst_attribute_name = params.extract_input<std::string>("Destination");

  if (src_attribute_name.empty() || dst_attribute_name.empty()) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  const NodeGeometryAttributeTransfer &storage =
      *(const NodeGeometryAttributeTransfer *)params.node().storage;
  const AttributeDomain dst_domain = (AttributeDomain)storage.domain;
  const GeometryNodeAttributeTransferMappingMode mapping =
      (GeometryNodeAttributeTransferMappingMode)storage.mapping;

  geometry_set = bke::geometry_set_realize_instances(geometry_set);
  target_geometry_set = bke::geometry_set_realize_instances(target_geometry_set);

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_transfer()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_ATTRIBUTE_TRANSFER, "Attribute Transfer", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(
      &ntype, geo_node_attribute_transfer_in, geo_node_attribute_transfer_out);
  node_type_init(&ntype, blender::nodes::geo_node_attribute_transfer_init);
  node_type_storage(&ntype,
                    "NodeGeometryAttributeTransfer",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_transfer_exec;
  ntype.draw_buttons = geo_node_attribute_transfer_layout;
  nodeRegisterType(&ntype);
}
