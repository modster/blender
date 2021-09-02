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

#ifdef WITH_OPENVDB
#  include <openvdb/tools/LevelSetSphere.h>
#endif

#include "BKE_lib_id.h"
#include "BKE_volume.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "DEG_depsgraph_query.h"

#include "node_geometry_util.hh"

namespace blender::nodes {

static void geo_node_level_set_primitive_sphere_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>("Radius").default_value(1.0f).min(0.0f).subtype(PROP_DISTANCE);
  b.add_input<decl::Vector>("Target").subtype(PROP_TRANSLATION);
  b.add_input<decl::Float>("Voxel Size").default_value(0.3f).min(0.01f).subtype(PROP_DISTANCE);
  b.add_output<decl::Geometry>("Level Set");
}

#ifdef WITH_OPENVDB

static Volume *level_set_primitive_sphere(GeoNodeExecParams &params)
{
  Volume *volume = (Volume *)BKE_id_new_nomain(ID_VO, nullptr);
  BKE_volume_init_grids(volume);

  const float3 center = params.get_input<float3>("Center");
  openvdb::FloatGrid::Ptr grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
      params.extract_input<float>("Radius"),
      openvdb::Vec3f(center.x, center.y, center.z),
      params.extract_input<float>("Voxel Size"));

  BKE_volume_grid_add_vdb(volume, "level_set", std::move(grid));

  return volume;
}

#endif /* WITH_OPENVDB */

static void geo_node_level_set_primitive_sphere_exec(GeoNodeExecParams params)
{
#ifdef WITH_OPENVDB
  Volume *volume = level_set_primitive_sphere(params);
  params.set_output("Level Set", GeometrySet::create_with_volume(volume));
#else
  params.set_output("Level Set", GeometrySet());
#endif
}

}  // namespace blender::nodes

void register_node_type_geo_level_set_primitive_sphere()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_LEVEL_SET_PRIMITIVE_SPHERE, "Level Set Sphere", NODE_CLASS_GEOMETRY, 0);
  ntype.declare = blender::nodes::geo_node_level_set_primitive_sphere_declare;
  ntype.geometry_node_execute = blender::nodes::geo_node_level_set_primitive_sphere_exec;

  nodeRegisterType(&ntype);
}
