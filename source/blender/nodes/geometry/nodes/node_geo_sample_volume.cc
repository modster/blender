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

#include "FN_generic_array.hh"

#include "NOD_type_conversions.hh"

#include "node_geometry_util.hh"

namespace blender::nodes {

static void geo_node_sample_volume_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>(N_("Volume"))
      .only_realized_data()
      .supported_type(GEO_COMPONENT_TYPE_VOLUME);

  b.add_input<decl::Vector>(N_("Position")).implicit_field();

  b.add_output<decl::Vector>(N_("Value")).dependent_field({1, 2, 3, 4, 5, 6});
  b.add_output<decl::Float>(N_("Value"), "Value_001").dependent_field({1, 2, 3, 4, 5, 6});
  b.add_output<decl::Bool>(N_("Value"), "Value_002").dependent_field({1, 2, 3, 4, 5, 6});
  b.add_output<decl::Int>(N_("Value"), "Value_003").dependent_field({1, 2, 3, 4, 5, 6});
}

static void geo_node_sample_volume_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "data_type", 0, "", ICON_NONE);
  uiItemR(layout, ptr, "mapping", 0, "", ICON_NONE);
}

static void geo_node_sample_volume_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometrySampleVolume *data = (NodeGeometrySampleVolume *)MEM_callocN(
      sizeof(NodeGeometrySampleVolume), __func__);
  data->interpolation = GEO_NODE_VOLUME_SAMPLE_LINEAR;
  data->data_type = CD_PROP_FLOAT;
  node->storage = data;
}

static void geo_node_sample_volume_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  const NodeGeometrySampleVolume &data = *(const NodeGeometrySampleVolume *)node->storage;
  const CustomDataType data_type = static_cast<CustomDataType>(data.data_type);

  bNodeSocket *out_socket_vector = (bNodeSocket *)BLI_findlink(&node->outputs, 4);
  bNodeSocket *out_socket_float = out_socket_vector->next;
  bNodeSocket *out_socket_color4f = out_socket_float->next;
  bNodeSocket *out_socket_boolean = out_socket_color4f->next;
  bNodeSocket *out_socket_int32 = out_socket_boolean->next;

  nodeSetSocketAvailability(out_socket_vector, data_type == CD_PROP_FLOAT3);
  nodeSetSocketAvailability(out_socket_float, data_type == CD_PROP_FLOAT);
  nodeSetSocketAvailability(out_socket_color4f, data_type == CD_PROP_COLOR);
  nodeSetSocketAvailability(out_socket_boolean, data_type == CD_PROP_BOOL);
  nodeSetSocketAvailability(out_socket_int32, data_type == CD_PROP_INT32);
}

#ifdef WITH_OPENVDB

using openvdb::GridBase;

static const CPPType *grid_type_to_sample_type(GridBase::ConstPtr grid)
{
  switch (BKE_volume_grid_type_openvdb(*grid)) {
    case VOLUME_GRID_MASK:
    case VOLUME_GRID_BOOLEAN:
      return &CPPType::get<bool>();
    case VOLUME_GRID_FLOAT:
      return &CPPType::get<float>();
    case VOLUME_GRID_INT:
      return &CPPType::get<int>();
    case VOLUME_GRID_VECTOR_FLOAT:
      return &CPPType::get<float3>();
    case VOLUME_GRID_UNKNOWN:
    case VOLUME_GRID_INT64:
    case VOLUME_GRID_STRING:
    case VOLUME_GRID_VECTOR_DOUBLE:
    case VOLUME_GRID_DOUBLE:
    case VOLUME_GRID_VECTOR_INT:
    case VOLUME_GRID_POINTS:
      return nullptr;
  }
  BLI_assert_unreachable();
  return nullptr;
}

static void sample_grid(GridBase::ConstPtr base_grid,
                        IndexMask mask,
                        Span<float3> positions,
                        GMutableSpan result)
{
  bke::volume::to_static_type(BKE_volume_grid_type_openvdb(*base_grid), [&](auto dummy) {
    using GridType = decltype(dummy);

    const GridType &grid = static_cast<const GridType &>(*base_grid);
    openvdb::tools::GridSampler<GridType, openvdb::tools::BoxSampler> sampler(grid);

    if constexpr (std::is_same_v<GridType, openvdb::FloatGrid>) {
      MutableSpan<float> dst = result.typed<float>();
      for (const int64_t i : mask) {
        dst[i] = sampler.wsSample({positions[i].x, positions[i].y, positions[i].z});
      }
    }
    else if constexpr (std::is_same_v<GridType, openvdb::Int32Grid>) {
      MutableSpan<int> dst = result.typed<int>();
      for (const int64_t i : mask) {
        dst[i] = sampler.wsSample({positions[i].x, positions[i].y, positions[i].z});
      }
    }
    else if constexpr (std::is_same_v<GridType, openvdb::BoolGrid> ||
                       std::is_same_v<GridType, openvdb::MaskGrid>) {
      MutableSpan<bool> dst = result.typed<bool>();
      for (const int64_t i : mask) {
        dst[i] = sampler.wsSample({positions[i].x, positions[i].y, positions[i].z});
      }
    }
    else if constexpr (std::is_same_v<GridType, openvdb::Vec3fGrid>) {
      MutableSpan<float3> dst = result.typed<float3>();
      for (const int64_t i : mask) {
        const openvdb::Vec3f value = sampler.wsSample(
            {positions[i].x, positions[i].y, positions[i].z});
        dst[i] = float3(value.x(), value.y(), value.z());
      }
    }
  });
}

static void combine_data_with_result(GSpan data, GMutableSpan result)
{
  BLI_assert(data.type() == result.type());
}

class SampleVolumeFunction : public fn::MultiFunction {
 private:
  GeometrySet geometry_set_;
  const CPPType *type_;

  fn::MFSignature signature_;

 public:
  SampleVolumeFunction(GeometrySet geometry_set, const CPPType &type)
      : geometry_set_(std::move(geometry_set)), type_(&type)
  {
    static fn::MFSignature signature = create_signature();
    signature_ = this->create_signature();
    this->set_signature(&signature_);
  }

  fn::MFSignature create_signature()
  {
    blender::fn::MFSignatureBuilder signature{"Sample Volume"};
    signature.single_input<float3>("Position");
    signature.single_output("Value", *type_);
    return signature.build();
  }

  void call(IndexMask mask, fn::MFParams params, fn::MFContext UNUSED(context)) const override
  {
    const VArray<float3> &positions = params.readonly_single_input<float3>(0, "Position");
    const VArray_Span<float3> positions_span{positions};
    GMutableSpan result = params.uninitialized_single_output(1, "Value");

    const VolumeComponent *component = geometry_set_.get_component_for_read<VolumeComponent>();
    const Volume *volume = component->get_for_read();

    for (const int i : IndexRange(BKE_volume_num_grids(volume))) {
      const VolumeGrid *volume_grid = BKE_volume_grid_get_for_read(volume, i);
      GridBase::ConstPtr grid = BKE_volume_grid_openvdb_for_read(volume, volume_grid);
      const CPPType *grid_type = grid_type_to_sample_type(grid);
      if (grid_type == nullptr) {
        continue;
      }
      if (grid_type == type_) {
        sample_grid(grid, mask, positions_span, result);
      }
      else {
        fn::GArray<> data_grid_type{*grid_type, mask.min_array_size()};
        sample_grid(grid, mask, positions_span, data_grid_type.as_mutable_span());
        fn::GArray<> data_converted{*grid_type, mask.min_array_size()};
        const DataTypeConversions &conversions = get_implicit_type_conversions();
        conversions.try_convert(data_grid_type.as_span(), data_converted.as_mutable_span(), mask);
        combine_data_with_result(data_converted, result);
      }
    }
  }
};

#endif

static void geo_node_sample_volume_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Curve");
  const bNode &node = params.node();
  const NodeGeometrySampleVolume &storage = *(const NodeGeometrySampleVolume *)node.storage;
  const CustomDataType data_type = static_cast<CustomDataType>(storage.data_type);

  auto return_default = [&]() {
    switch (data_type) {
      case CD_PROP_FLOAT3:
        params.set_output("Value", fn::make_constant_field<float3>({0.0f, 0.0f, 0.0f}));
        break;
      case CD_PROP_FLOAT:
        params.set_output("Value_001", fn::make_constant_field<float>(0.0f));
        break;
      case CD_PROP_BOOL:
        params.set_output("Value_002", fn::make_constant_field<bool>(false));
        break;
      case CD_PROP_INT32:
        params.set_output("Value_003", fn::make_constant_field<int>(0));
        break;
      default:
        BLI_assert_unreachable();
    }
  };

#ifdef WITH_OPENVDB

  const VolumeComponent *component = geometry_set.get_component_for_read<VolumeComponent>();
  if (component == nullptr) {
    return return_default();
  }

  const Volume *volume = component->get_for_read();
  if (volume == nullptr) {
    return return_default();
  }

  const Main *bmain = DEG_get_bmain(params.depsgraph());
  BKE_volume_load(volume, bmain);

  auto sample_fn = std::make_unique<SampleVolumeFunction>(
      std::move(geometry_set), bke::custom_data_type_to_cpp_type(data_type));
  auto sample_op = std::make_shared<FieldOperation>(
      FieldOperation(std::move(sample_fn), {params.get_input<Field<float3>>("Position")}));

  switch (data_type) {
    case CD_PROP_FLOAT3:
      params.set_output("Value", Field<float3>(sample_op));
      break;
    case CD_PROP_FLOAT:
      params.set_output("Value_001", Field<float>(sample_op));
      break;
    case CD_PROP_BOOL:
      params.set_output("Value_002", Field<bool>(sample_op));
      break;
    case CD_PROP_INT32:
      params.set_output("Value_003", Field<int>(sample_op));
      break;
    default:
      BLI_assert_unreachable();
  }
#else

#endif
}

}  // namespace blender::nodes

void register_node_type_geo_sample_volume()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_SAMPLE_VOLUME, "Sample Volume", NODE_CLASS_GEOMETRY, 0);
  ntype.declare = blender::nodes::geo_node_sample_volume_declare;
  ntype.geometry_node_execute = blender::nodes::geo_node_sample_volume_exec;
  nodeRegisterType(&ntype);
}
