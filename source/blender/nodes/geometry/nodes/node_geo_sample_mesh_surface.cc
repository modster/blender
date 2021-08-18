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

#include "BLI_kdopbvh.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_attribute_math.hh"
#include "BKE_bvhutils.h"
#include "BKE_customdata.h"
#include "BKE_mesh_runtime.h"
#include "BKE_mesh_sample.hh"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_sample_mesh_surface_in[] = {
    {SOCK_GEOMETRY, N_("Mesh")},
    {SOCK_VECTOR,
     N_("Position"),
     0.0f,
     0.0f,
     0.0f,
     1.0f,
     -FLT_MAX,
     FLT_MAX,
     PROP_TRANSLATION,
     SOCK_HIDE_VALUE | SOCK_FIELD},
    {SOCK_RGBA, N_("Custom"), 1, 1, 1, 1, 0, 1, PROP_NONE, SOCK_FIELD},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_sample_mesh_surface_out[] = {
    {SOCK_VECTOR, N_("Position")},
    {SOCK_VECTOR, N_("Normal")},
    {SOCK_FLOAT, N_("Distance")},
    {SOCK_RGBA, N_("Custom")},
    {-1, ""},
};

namespace blender::nodes {

class SampleMeshSurfaceFunction : public fn::MultiFunction {
 private:
  GeometrySet geometry_set_;
  AnonymousCustomDataLayerID *attribute_id_;

 public:
  SampleMeshSurfaceFunction(GeometrySet geometry_set, AnonymousCustomDataLayerID *attribute_id)
      : geometry_set_(std::move(geometry_set)), attribute_id_(attribute_id)
  {
    static fn::MFSignature signature = create_signature();
    this->set_signature(&signature);
    CustomData_anonymous_id_strong_increment(attribute_id_);
  }

  ~SampleMeshSurfaceFunction() override
  {
    CustomData_anonymous_id_strong_decrement(attribute_id_);
  }

  static blender::fn::MFSignature create_signature()
  {
    blender::fn::MFSignatureBuilder signature{"Sample Mesh Surface"};
    signature.single_input<float3>("Position");
    signature.single_output<float3>("Position");
    signature.single_output<float3>("Normal");
    signature.single_output<float>("Distance");
    signature.single_output<ColorGeometry4f>("Custom");
    return signature.build();
  }

  void call(IndexMask mask, fn::MFParams params, fn::MFContext UNUSED(context)) const override
  {
    const VArray<float3> &src_positions = params.readonly_single_input<float3>(0, "Position");
    MutableSpan<float3> sampled_positions = params.uninitialized_single_output<float3>(1,
                                                                                       "Position");
    MutableSpan<float3> sampled_normals = params.uninitialized_single_output<float3>(2, "Normal");
    MutableSpan<float> sampled_distances = params.uninitialized_single_output<float>(3,
                                                                                     "Distance");
    MutableSpan<ColorGeometry4f> sampled_custom =
        params.uninitialized_single_output<ColorGeometry4f>(4, "Custom");

    auto return_default = [&]() {
      sampled_positions.fill_indices(mask, {0, 0, 0});
      sampled_normals.fill_indices(mask, {0, 0, 0});
      sampled_distances.fill_indices(mask, 0.0f);
      sampled_custom.fill_indices(mask, {0, 0, 0, 1});
    };

    if (!geometry_set_.has_mesh()) {
      return return_default();
    }

    const MeshComponent *mesh_component = geometry_set_.get_component_for_read<MeshComponent>();
    const Mesh *mesh = mesh_component->get_for_read();

    GVArrayPtr attribute_ptr = mesh_component->attribute_try_get_anonymous_for_read(
        *attribute_id_, ATTR_DOMAIN_CORNER, CD_PROP_COLOR, nullptr);
    if (!attribute_ptr) {
      return return_default();
    }
    GVArray_Typed<ColorGeometry4f> attribute{*attribute_ptr};

    const MLoopTri *looptris = BKE_mesh_runtime_looptri_ensure(mesh);

    BVHTreeFromMesh tree_data;
    BKE_bvhtree_from_mesh_get(&tree_data, mesh, BVHTREE_FROM_LOOPTRI, 2);

    for (const int i : mask) {
      BVHTreeNearest nearest;
      nearest.dist_sq = FLT_MAX;
      const float3 src_position = src_positions[i];
      BLI_bvhtree_find_nearest(
          tree_data.tree, src_position, &nearest, tree_data.nearest_callback, &tree_data);
      sampled_positions[i] = nearest.co;
      sampled_normals[i] = nearest.no;
      sampled_distances[i] = sqrtf(nearest.dist_sq);

      const MLoopTri &looptri = looptris[nearest.index];

      float3 v1 = mesh->mvert[mesh->mloop[looptri.tri[0]].v].co;
      float3 v2 = mesh->mvert[mesh->mloop[looptri.tri[1]].v].co;
      float3 v3 = mesh->mvert[mesh->mloop[looptri.tri[2]].v].co;

      ColorGeometry4f col1 = attribute[looptri.tri[0]];
      ColorGeometry4f col2 = attribute[looptri.tri[1]];
      ColorGeometry4f col3 = attribute[looptri.tri[2]];

      float3 bary_coords;
      interp_weights_tri_v3(bary_coords, v1, v2, v3, nearest.co);
      ColorGeometry4f final_col = attribute_math::mix3(bary_coords, col1, col2, col3);
      sampled_custom[i] = final_col;
    }
  }
};

static void geo_node_sample_mesh_surface_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh");
  geometry_set = geometry_set_realize_instances(geometry_set);

  FieldPtr position_field = params.get_input_field<float3>("Position").field();
  bke::FieldRef<ColorGeometry4f> attribute_field = params.get_input_field<ColorGeometry4f>(
      "Custom");

  AnonymousCustomDataLayerID *layer_id = CustomData_anonymous_id_new("Sample Mesh Surface");
  MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
  try_freeze_field_on_geometry(
      mesh_component, *layer_id, ATTR_DOMAIN_CORNER, *attribute_field.field());

  auto make_output_field = [&](int out_param_index) -> FieldPtr {
    auto fn = std::make_unique<SampleMeshSurfaceFunction>(geometry_set, layer_id);
    return new bke::MultiFunctionField(Vector<FieldPtr>{position_field},
                                       optional_ptr<const fn::MultiFunction>{std::move(fn)},
                                       out_param_index);
  };

  params.set_output("Position", bke::FieldRef<float3>(make_output_field(1)));
  params.set_output("Normal", bke::FieldRef<float3>(make_output_field(2)));
  params.set_output("Distance", bke::FieldRef<float>(make_output_field(3)));
  params.set_output("Custom", bke::FieldRef<ColorGeometry4f>(make_output_field(4)));
}

}  // namespace blender::nodes

void register_node_type_geo_sample_mesh_surface()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_SAMPLE_MESH_SURFACE, "Sample Mesh Surface", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(
      &ntype, geo_node_sample_mesh_surface_in, geo_node_sample_mesh_surface_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_sample_mesh_surface_exec;
  nodeRegisterType(&ntype);
}
