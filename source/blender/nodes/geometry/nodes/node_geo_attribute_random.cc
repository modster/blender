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

#include "BLI_rand.hh"

#include "DNA_customdata_types.h"
#include "DNA_windowmanager_types.h"

#include "BKE_attribute.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_attribute_random_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Attribute")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_random_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_attribute_random_init(bNodeTree *UNUSED(tree), bNode *node)
{
  node->custom1 = CD_PROP_FLOAT;
}

namespace blender::nodes {
static void geo_attribute_random_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  std::string attribute_name = params.extract_input<std::string>("Attribute");

  RandomNumberGenerator rng(0);
  CustomDataType data_type = static_cast<CustomDataType>(params.node().custom1);
  AttributeDomain domain = static_cast<AttributeDomain>(params.node().custom2);

  if (geometry_set.has_mesh()) {
    Mesh *mesh = geometry_set.get_mesh_for_write();
  }

  if (geometry_set.has_pointcloud()) {
    PointCloud *point_cloud = geometry_set.get_pointcloud_for_write();
  }

  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_attribute_random()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_ATTRIBUTE_RANDOM, "Random Attribute", 0, 0);
  node_type_socket_templates(&ntype, geo_node_attribute_random_in, geo_node_attribute_random_out);
  node_type_init(&ntype, geo_attribute_random_init);
  ntype.geometry_node_execute = blender::nodes::geo_attribute_random_exec;
  nodeRegisterType(&ntype);
}
