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
#  include <openvdb/openvdb.h>
#  include <openvdb/tools/GridTransformer.h>
#  include <openvdb/tools/LevelSetMorph.h>
#endif

#include "BKE_volume.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "DEG_depsgraph_query.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_level_set_morph_in[] = {
    {SOCK_GEOMETRY, N_("Source")},
    {SOCK_GEOMETRY, N_("Target")},
    {SOCK_FLOAT, N_("Factor"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, PROP_UNSIGNED},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_level_set_morph_out[] = {
    {SOCK_GEOMETRY, N_("Result")},
    {-1, ""},
};

static void geo_node_level_set_morph_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "spatial_scheme", 0, "", ICON_NONE);
  uiItemR(layout, ptr, "temporal_scheme", 0, "", ICON_NONE);
}

static void geo_node_level_set_morph_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryLevelSetMorph *data = (NodeGeometryLevelSetMorph *)MEM_callocN(
      sizeof(NodeGeometryLevelSetMorph), __func__);
  data->spatial_scheme = GEO_NODE_LEVEL_SET_MORPH_SPATIAL_HJWENO5;
  data->temporal_scheme = GEO_NODE_LEVEL_SET_MORPH_SPATIAL_2ND;
  node->storage = data;
}

namespace blender::nodes {

#ifdef WITH_OPENVDB

static openvdb::math::TemporalIntegrationScheme temporal_scheme_to_openvdb(
    const GeometryNodeLevelSetTemporalScheme value)
{
  switch (value) {
    case GEO_NODE_LEVEL_SET_MORPH_SPATIAL_FORWARD_EULER:
      return openvdb::math::TVD_RK1;
    case GEO_NODE_LEVEL_SET_MORPH_SPATIAL_2ND:
      return openvdb::math::TVD_RK2;
    case GEO_NODE_LEVEL_SET_MORPH_SPATIAL_3RD:
      return openvdb::math::TVD_RK3;
  }
  BLI_assert_unreachable();
  return openvdb::math::TVD_RK1;
}

static openvdb::math::BiasedGradientScheme spatial_scheme_to_openvdb(
    const GeometryNodeLevelSetSpatialScheme value)
{
  switch (value) {
    case GEO_NODE_LEVEL_SET_MORPH_SPATIAL_FIRST:
      return openvdb::math::FIRST_BIAS;
    case GEO_NODE_LEVEL_SET_MORPH_SPATIAL_HJWENO5:
      return openvdb::math::HJWENO5_BIAS;
  }
  BLI_assert_unreachable();
  return openvdb::math::FIRST_BIAS;
}

static void level_set_morph(Volume &volume_a,
                            const Volume &volume_b,
                            const float factor,
                            const openvdb::math::BiasedGradientScheme spatial_scheme,
                            const openvdb::math::TemporalIntegrationScheme temporal_scheme,
                            const GeoNodeExecParams &params)
{
  VolumeGrid *volume_grid_a = BKE_volume_grid_get_for_write(&volume_a, 0);
  const VolumeGrid *volume_grid_b = BKE_volume_grid_get_for_read(&volume_b, 0);
  if (ELEM(nullptr, volume_grid_a, volume_grid_b)) {
    if (volume_grid_a == nullptr) {
      params.error_message_add(NodeWarningType::Info, TIP_("Volume 1 is empty"));
    }
    if (volume_grid_b == nullptr) {
      params.error_message_add(NodeWarningType::Info, TIP_("Volume 2 is empty"));
    }
    return;
  }
  openvdb::GridBase::Ptr grid_base_a = BKE_volume_grid_openvdb_for_write(&volume_a, volume_grid_a);
  openvdb::GridBase::ConstPtr grid_base_b = BKE_volume_grid_openvdb_for_read(&volume_b,
                                                                             volume_grid_b);
  if (grid_base_a->getGridClass() != openvdb::GridClass::GRID_LEVEL_SET ||
      grid_base_b->getGridClass() != openvdb::GridClass::GRID_LEVEL_SET) {
    if (grid_base_a->getGridClass() != openvdb::GridClass::GRID_LEVEL_SET) {
      params.error_message_add(NodeWarningType::Error, TIP_("Volume 1 is not a level set"));
    }
    if (grid_base_b->getGridClass() != openvdb::GridClass::GRID_LEVEL_SET) {
      params.error_message_add(NodeWarningType::Error, TIP_("Volume 2 is not a level set"));
    }
    return;
  }

  const VolumeGridType grid_type_a = BKE_volume_grid_type(volume_grid_a);
  const VolumeGridType grid_type_b = BKE_volume_grid_type(volume_grid_b);
  if (grid_type_a != grid_type_b) {
    params.error_message_add(NodeWarningType::Error, TIP_("Volume grid types do not match"));
    return;
  }

  const bool needs_resample = grid_base_a->transform() != grid_base_b->transform();

  bke::volume::to_static_type(grid_type_a, [&](auto dummy) {
    using GridType = decltype(dummy);
    if constexpr (std::is_same_v<GridType, openvdb::FloatGrid>) {
      GridType &grid_a = static_cast<GridType &>(*grid_base_a);
      const GridType &grid_b = static_cast<const GridType &>(*grid_base_b);

      typename GridType::Ptr grid_b_resampled;
      if (needs_resample) {
        grid_b_resampled = GridType::create();
        grid_b_resampled->setTransform(grid_base_a->constTransform().copy());

        openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(grid_b, *grid_b_resampled);
        openvdb::tools::pruneLevelSet(grid_b_resampled->tree());
      }

      openvdb::tools::LevelSetMorphing<GridType> morph(
          grid_a, needs_resample ? *grid_b_resampled : grid_b);
      morph.setTemporalScheme(temporal_scheme);
      morph.setSpatialScheme(spatial_scheme);
      morph.advect(0.0f, factor / grid_a.voxelSize().length());
    }
  });
}

#endif /* WITH_OPENVDB */

static void geo_node_level_set_morph_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set_a = params.extract_input<GeometrySet>("Source");
#ifdef WITH_OPENVDB
  GeometrySet geometry_set_b = params.extract_input<GeometrySet>("Target");

  SCOPED_TIMER(__func__);

  const NodeGeometryLevelSetMorph &storage =
      *(const NodeGeometryLevelSetMorph *)params.node().storage;
  const GeometryNodeLevelSetSpatialScheme spatial_scheme = (GeometryNodeLevelSetSpatialScheme)
                                                               storage.spatial_scheme;
  const GeometryNodeLevelSetTemporalScheme temporal_scheme = (GeometryNodeLevelSetTemporalScheme)
                                                                 storage.temporal_scheme;

  Volume *volume_a = geometry_set_a.get_volume_for_write();
  const Volume *volume_b = geometry_set_b.get_volume_for_read();
  if (volume_a == nullptr || volume_b == nullptr) {
    params.set_output("Result", std::move(geometry_set_a));
    return;
  }

  const Main *bmain = DEG_get_bmain(params.depsgraph());
  BKE_volume_load(volume_a, bmain);
  BKE_volume_load(volume_b, bmain);

  level_set_morph(*volume_a,
                  *volume_b,
                  params.extract_input<float>("Factor"),
                  spatial_scheme_to_openvdb(spatial_scheme),
                  temporal_scheme_to_openvdb(temporal_scheme),
                  params);
#else
  params.error_message_add(NodeWarningType::Error,
                           TIP_("Disabled, Blender was compiled without OpenVDB"));
#endif

  params.set_output("Result", std::move(geometry_set_a));
}

}  // namespace blender::nodes

void register_node_type_geo_level_set_morph()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_LEVEL_SET_MORPH, "Level Set Morph", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_level_set_morph_in, geo_node_level_set_morph_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_level_set_morph_exec;
  node_type_storage(
      &ntype, "NodeGeometryLevelSetMorph", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, geo_node_level_set_morph_init);
  ntype.draw_buttons = geo_node_level_set_morph_layout;

  nodeRegisterType(&ntype);
}
