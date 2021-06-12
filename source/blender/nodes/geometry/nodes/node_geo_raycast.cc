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

static bNodeSocketTemplate geo_node_raycast_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_GEOMETRY, N_("Source Geometry")},
    {SOCK_STRING, N_("Ray Origin")},
    {SOCK_STRING, N_("Ray Direction")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_raycast_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_raycast_layout(uiLayout *layout,
                                               bContext *UNUSED(C),
                                               PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  //uiItemR(layout, ptr, "domain", 0, IFACE_("Domain"), ICON_NONE);
  //uiItemR(layout, ptr, "mapping", 0, IFACE_("Mapping"), ICON_NONE);
}

static void geo_node_raycast_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryRaycast *data = (NodeGeometryRaycast *)MEM_callocN(
      sizeof(NodeGeometryRaycast), __func__);

  node->storage = data;
}

namespace blender::nodes {

static void geo_node_raycast_exec(GeoNodeExecParams params)
{
  GeometrySet dst_geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet src_geometry_set = params.extract_input<GeometrySet>("Source Geometry");
  const std::string src_attribute_name = params.extract_input<std::string>("Ray Origin");
  const std::string dst_attribute_name = params.extract_input<std::string>("Ray Direction");

  if (src_attribute_name.empty() || dst_attribute_name.empty()) {
    params.set_output("Geometry", dst_geometry_set);
    return;
  }

  dst_geometry_set = bke::geometry_set_realize_instances(dst_geometry_set);
  src_geometry_set = bke::geometry_set_realize_instances(src_geometry_set);

  //if (dst_geometry_set.has<MeshComponent>()) {
  //  transfer_attribute(params,
  //                     src_geometry_set,
  //                     dst_geometry_set.get_component_for_write<MeshComponent>(),
  //                     src_attribute_name,
  //                     dst_attribute_name);
  //}
  //if (dst_geometry_set.has<PointCloudComponent>()) {
  //  transfer_attribute(params,
  //                     src_geometry_set,
  //                     dst_geometry_set.get_component_for_write<PointCloudComponent>(),
  //                     src_attribute_name,
  //                     dst_attribute_name);
  //}

  params.set_output("Geometry", dst_geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_raycast()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_RAYCAST, "Raycast", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(
      &ntype, geo_node_raycast_in, geo_node_raycast_out);
  node_type_init(&ntype, geo_node_raycast_init);
  node_type_storage(&ntype,
                    "NodeGeometryRaycast",
                    node_free_standard_storage,
                    node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_raycast_exec;
  ntype.draw_buttons = geo_node_raycast_layout;
  nodeRegisterType(&ntype);
}
