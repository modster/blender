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

#include "BLI_kdopbvh.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_pointcloud_types.h"

#include "BKE_bvhutils.h"
#include "BKE_mesh_runtime.h"
#include "BKE_mesh_sample.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_attribute_get_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Name")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_get_out[] = {
    {SOCK_ATTRIBUTE, N_("Attribute")},
    {-1, ""},
};

static void geo_node_attribute_get_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "domain", 0, IFACE_("Domain"), ICON_NONE);
  uiItemR(layout, ptr, "data_type", 0, "", ICON_NONE);
}

static void geo_node_attribute_get_init(bNodeTree *UNUSED(tree), bNode *node)
{
  node->custom1 = CD_PROP_FLOAT;
  node->custom2 = ATTR_DOMAIN_AUTO;

  nodeFindSocket(node, SOCK_OUT, "Attribute")->display_shape = SOCK_DISPLAY_SHAPE_SQUARE;
}

namespace blender::nodes {

static void geo_node_attribute_get_exec(GeoNodeExecParams params)
{
  params.set_output("Attribute", AttributeRef::None);

  const CustomDataType data_type = (CustomDataType)params.node().custom1;
  const AttributeDomain domain = (AttributeDomain)params.node().custom2;
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  const std::string attribute_name = params.extract_input<std::string>("Name");

  if (attribute_name.empty()) {
    params.set_output("Attribute", AttributeRef::None);
    return;
  }

  AttributeRef attribute = AttributeRef(attribute_name, domain, data_type);

  /* TODO check for existence of the attribute on the geometry.
   * This isn't really necessary for it to function, but can help catch invalid
   * references early in the node graph.
   * Technically this node does not even need a geometry input other than for
   * checking validity. It could also just create an AttributeRef on good faith.
   */

  params.set_output("Attribute", attribute);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_get()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_ATTRIBUTE_GET, "Attribute Get", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_attribute_get_in, geo_node_attribute_get_out);
  node_type_init(&ntype, geo_node_attribute_get_init);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_get_exec;
  ntype.draw_buttons = geo_node_attribute_get_layout;
  nodeRegisterType(&ntype);
}
