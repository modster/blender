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

#include "DEG_depsgraph_query.h"
#ifdef WITH_OPENVDB
#  include <openvdb/tools/GridTransformer.h>
#  include <openvdb/tools/VolumeToMesh.h>
#endif

#include "BKE_lib_id.h"
#include "BKE_volume.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_volume_sample_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_GEOMETRY, N_("Volume")},
    {SOCK_STRING, N_("Grid")},
    {SOCK_STRING, N_("Attribute")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_volume_sample_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

#ifdef WITH_OPENVDB

static std::optional<CustomDataType> grid_type_to_attribute_type(openvdb::GridBase::ConstPtr grid)
{
  switch (BKE_volume_grid_type_openvdb(*grid)) {
    case VOLUME_GRID_MASK:
    case VOLUME_GRID_BOOLEAN:
      return CD_PROP_BOOL;
    case VOLUME_GRID_FLOAT:
      return CD_PROP_FLOAT;
    case VOLUME_GRID_INT:
      return CD_PROP_INT32;
    case VOLUME_GRID_VECTOR_FLOAT:
      return CD_PROP_FLOAT3;
    case VOLUME_GRID_UNKNOWN:
    case VOLUME_GRID_INT64:
    case VOLUME_GRID_STRING:
    case VOLUME_GRID_VECTOR_DOUBLE:
    case VOLUME_GRID_DOUBLE:
    case VOLUME_GRID_VECTOR_INT:
    case VOLUME_GRID_POINTS:
      return {};
  }
  BLI_assert_unreachable();
  return {};
}

static void execute_on_component(GeometryComponent &component,
                                 const StringRef attribute_name,
                                 const CustomDataType data_type,
                                 openvdb::GridBase::ConstPtr base_grid)
{
  OutputAttribute attribute = component.attribute_try_get_for_output_only(
      attribute_name, ATTR_DOMAIN_POINT, data_type);
  if (!attribute) {
    return;
  }
  GVArray_Typed<float3> positions = component.attribute_get_for_read<float3>(
      "position", ATTR_DOMAIN_POINT, {0, 0, 0});

  bke::volume::to_static_type(BKE_volume_grid_type_openvdb(*base_grid), [&](auto dummy) {
    using GridType = decltype(dummy);

    const GridType &grid = static_cast<const GridType &>(*base_grid);
    openvdb::tools::GridSampler<GridType, openvdb::tools::BoxSampler> sampler(grid);

    if constexpr (std::is_same_v<GridType, openvdb::FloatGrid>) {
      GVMutableArray_Typed<float> attribute_typed(*attribute);
      VMutableArray<float> &result_varray = *attribute_typed;
      for (const int i : positions.index_range()) {
        const float value = sampler.wsSample({positions[i].x, positions[i].y, positions[i].z});
        result_varray.set(i, value);
      }
    }
    else if constexpr (std::is_same_v<GridType, openvdb::Int32Grid>) {
      GVMutableArray_Typed<int> attribute_typed(*attribute);
      VMutableArray<int> &result_varray = *attribute_typed;
      for (const int i : positions.index_range()) {
        const int value = sampler.wsSample({positions[i].x, positions[i].y, positions[i].z});
        result_varray.set(i, value);
      }
    }
    else if constexpr (std::is_same_v<GridType, openvdb::BoolGrid> ||
                       std::is_same_v<GridType, openvdb::MaskGrid>) {
      GVMutableArray_Typed<bool> attribute_typed(*attribute);
      VMutableArray<bool> &result_varray = *attribute_typed;
      for (const int i : positions.index_range()) {
        const openvdb::Vec3d position({positions[i].x, positions[i].y, positions[i].z});
        const bool value = sampler.wsSample({positions[i].x, positions[i].y, positions[i].z});
        result_varray.set(i, value);
      }
    }
    else if constexpr (std::is_same_v<GridType, openvdb::Vec3fGrid>) {
      GVMutableArray_Typed<float3> attribute_typed(*attribute);
      VMutableArray<float3> &result_varray = *attribute_typed;
      for (const int i : positions.index_range()) {
        const openvdb::Vec3f value = sampler.wsSample(
            {positions[i].x, positions[i].y, positions[i].z});
        result_varray.set(i, float3(value.x(), value.y(), value.z()));
      }
    }
  });

  attribute.save();
}

#endif /* WITH_OPENVDB */

static void geo_node_volume_sample_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = bke::geometry_set_realize_instances(geometry_set);

#ifdef WITH_OPENVDB
  GeometrySet volume_geometry_set = params.extract_input<GeometrySet>("Volume");

  const Volume *volume = volume_geometry_set.get_volume_for_read();
  if (volume == nullptr) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  const Main *bmain = DEG_get_bmain(params.depsgraph());
  BKE_volume_load(volume, bmain);

  const std::string grid_name = params.extract_input<std::string>("Grid");
  const VolumeGrid *volume_grid = BKE_volume_grid_find_for_read(volume, grid_name.c_str());
  if (volume_grid == nullptr) {
    params.set_output("Geometry", geometry_set);
    return;
  }

  openvdb::GridBase::ConstPtr grid = BKE_volume_grid_openvdb_for_read(volume, volume_grid);

  const std::optional<CustomDataType> data_type = grid_type_to_attribute_type(grid);
  if (!data_type) {
    params.error_message_add(NodeWarningType::Error, TIP_("Volume grid type not supported"));
    params.set_output("Geometry", geometry_set);
    return;
  }

  const std::string attribute_name = params.extract_input<std::string>("Attribute");

  static const Array<GeometryComponentType> types = {
      GEO_COMPONENT_TYPE_MESH, GEO_COMPONENT_TYPE_POINT_CLOUD, GEO_COMPONENT_TYPE_CURVE};
  for (const GeometryComponentType type : types) {
    if (geometry_set.has(type)) {
      execute_on_component(
          geometry_set.get_component_for_write(type), attribute_name, *data_type, grid);
    }
  }
#endif

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_volume_sample()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_VOLUME_SAMPLE, "Volume Sample", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_volume_sample_in, geo_node_volume_sample_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_volume_sample_exec;
  nodeRegisterType(&ntype);
}
