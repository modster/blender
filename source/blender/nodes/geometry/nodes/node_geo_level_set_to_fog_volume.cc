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
#  include <openvdb/tools/LevelSetUtil.h>
#endif

#include "BKE_volume.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "DEG_depsgraph_query.h"

#include "node_geometry_util.hh"

namespace blender::nodes {

static void geo_node_level_set_to_fog_volume_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Level Set");
  b.add_input<decl::Float>("Density").default_value(1.0f).min(0.0f);
  b.add_output<decl::Geometry>("Fog Volume");
}

#ifdef WITH_OPENVDB

static void level_set_to_fog_volume(Volume &volume, const GeoNodeExecParams &params)
{
  VolumeGrid *volume_grid = BKE_volume_grid_get_for_write(&volume, 0);
  if (volume_grid == nullptr) {
    params.error_message_add(NodeWarningType::Error, TIP_("Volume is empty"));
    return;
  }

  openvdb::GridBase::Ptr grid_base = BKE_volume_grid_openvdb_for_write(&volume, volume_grid);
  if (grid_base->getGridClass() != openvdb::GridClass::GRID_LEVEL_SET) {
    params.error_message_add(NodeWarningType::Error, TIP_("Volume is not a level set"));
  }

  bke::volume::to_static_type(BKE_volume_grid_type(volume_grid), [&](auto dummy) {
    using GridType = decltype(dummy);
    if constexpr (std::is_same_v<GridType, openvdb::FloatGrid>) {
      GridType &grid = static_cast<GridType &>(*grid_base);
      openvdb::tools::sdfToFogVolume(grid);

      const float density = params.get_input<float>("Density");
      if (density != 1.0f) {
        openvdb::tools::foreach (grid.beginValueOn(),
                                 [&](const openvdb::FloatGrid::ValueOnIter &iter) {
                                   iter.modifyValue([&](float &value) { value *= density; });
                                 });
      }
    }
  });
}

#endif /* WITH_OPENVDB */

static void geo_node_level_set_to_fog_volume_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Level Set");

#ifdef WITH_OPENVDB
  Volume *volume = geometry_set.get_volume_for_write();

  const Main *bmain = DEG_get_bmain(params.depsgraph());
  BKE_volume_load(volume, bmain);

  if (volume == nullptr) {
    params.set_output("Level Set", std::move(geometry_set));
    return;
  }

  level_set_to_fog_volume(*volume, params);
#endif

  params.set_output("Fog Volume", std::move(geometry_set));
}

}  // namespace blender::nodes

void register_node_type_geo_level_set_to_fog_volume()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_LEVEL_SET_TO_FOG_VOLUME, "Level Set to Fog Volume", NODE_CLASS_GEOMETRY, 0);
  ntype.declare = blender::nodes::geo_node_level_set_to_fog_volume_declare;
  ntype.geometry_node_execute = blender::nodes::geo_node_level_set_to_fog_volume_exec;

  nodeRegisterType(&ntype);
}
