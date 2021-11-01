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

#include "BKE_lib_id.h"
#include "BKE_volume.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "DEG_depsgraph_query.h"

#include "node_geometry_util.hh"

namespace blender::nodes {

static void geo_node_level_set_to_mask_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Level Set");
  b.add_output<decl::Geometry>("Mask Volume");
}

#ifdef WITH_OPENVDB

static Volume *level_set_to_mask(const Volume &volume, const GeoNodeExecParams &params)
{
  if (BKE_volume_num_grids(&volume) == 0) {
    params.error_message_add(NodeWarningType::Error, TIP_("Volume is empty"));
    return nullptr;
  }

  Volume *mask_volume = (Volume *)BKE_id_new_nomain(ID_VO, nullptr);
  BKE_volume_init_grids(mask_volume);

  for (const int i : IndexRange(BKE_volume_num_grids(&volume))) {
    const VolumeGrid *volume_grid = BKE_volume_grid_get_for_read(&volume, i);

    openvdb::GridBase::ConstPtr grid_base = BKE_volume_grid_openvdb_for_read(&volume, volume_grid);
    if (grid_base->getGridClass() != openvdb::GridClass::GRID_LEVEL_SET) {
      params.error_message_add(NodeWarningType::Error,
                               std::string(TIP_("Volume grid is not a level set: ")) +
                                   BKE_volume_grid_name(volume_grid));
    }

    bke::volume::to_static_type(BKE_volume_grid_type(volume_grid), [&](auto dummy) {
      using GridType = decltype(dummy);
      if constexpr (std::is_same_v<GridType, openvdb::FloatGrid>) {
        const GridType &grid = static_cast<const GridType &>(*grid_base);
        openvdb::BoolGrid::Ptr mask_grid = openvdb::tools::sdfInteriorMask(grid);
        BKE_volume_grid_add_vdb(*mask_volume, "mask_" + i, std::move(mask_grid));
      }
    });
  }

  return mask_volume;
}

#endif /* WITH_OPENVDB */

static void geo_node_level_set_to_mask_exec(GeoNodeExecParams params)
{
#ifdef WITH_OPENVDB
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Level Set");

  geometry_set.modify_geometry_sets([&](GeometrySet &geometry_set) {
    const Volume *volume = geometry_set.get_volume_for_read();

    if (volume == nullptr) {
      params.set_output("Mask Volume", std::move(geometry_set));
      return;
    }

    const Main *bmain = DEG_get_bmain(params.depsgraph());
    BKE_volume_load(volume, bmain);

    Volume *mask_volume = level_set_to_mask(*volume, params);
    geometry_set.replace_volume(mask_volume);
  });

  params.set_output("Mask Volume", geometry_set);
#else
  params.set_output("Mask Volume", GeometrySet());
#endif
}

}  // namespace blender::nodes

void register_node_type_geo_level_set_to_mask()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_LEVEL_SET_TO_MASK, "Level Set to Mask", NODE_CLASS_GEOMETRY, 0);
  ntype.declare = blender::nodes::geo_node_level_set_to_mask_declare;
  ntype.geometry_node_execute = blender::nodes::geo_node_level_set_to_mask_exec;

  nodeRegisterType(&ntype);
}
