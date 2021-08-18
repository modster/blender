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

#include "DNA_mesh_types.h"

#include "BKE_bvhutils.h"
#include "BKE_mesh_sample.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_raycast_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_GEOMETRY, N_("Target Geometry")},
    {SOCK_VECTOR,
     N_("Ray Direction"),
     0.0,
     0.0,
     1.0,
     0.0,
     -FLT_MAX,
     FLT_MAX,
     PROP_NONE,
     SOCK_FIELD},
    {SOCK_FLOAT, N_("Ray Length"), 100.0, 0.0, 0.0, 0.0, 0.0f, FLT_MAX, PROP_DISTANCE, SOCK_FIELD},
    {SOCK_STRING, N_("Target Attribute")},
    {SOCK_STRING, N_("Hit Attribute")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_raycast_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_BOOLEAN, N_("Is Hit")},
    {SOCK_VECTOR, N_("Hit Position")},
    {SOCK_VECTOR, N_("Hit Normal")},
    {SOCK_FLOAT, N_("Hit Distance")},
    {-1, ""},
};

static void geo_node_raycast_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "mapping", 0, IFACE_("Mapping"), ICON_NONE);
}

static void geo_node_raycast_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryRaycast *data = (NodeGeometryRaycast *)MEM_callocN(sizeof(NodeGeometryRaycast),
                                                                 __func__);
  node->storage = data;
}

namespace blender::nodes {

static void raycast_to_mesh(const Mesh *mesh,
                            const VArray<float3> &ray_origins,
                            const VArray<float3> &ray_directions,
                            const VArray<float> &ray_lengths,
                            const MutableSpan<bool> r_hit,
                            const MutableSpan<int> r_hit_indices,
                            const MutableSpan<float3> r_hit_positions,
                            const MutableSpan<float3> r_hit_normals,
                            const MutableSpan<float> r_hit_distances)
{
  BLI_assert(ray_origins.size() == ray_directions.size());
  BLI_assert(ray_origins.size() == ray_lengths.size());
  BLI_assert(ray_origins.size() == r_hit.size() || r_hit.is_empty());
  BLI_assert(ray_origins.size() == r_hit_indices.size() || r_hit_indices.is_empty());
  BLI_assert(ray_origins.size() == r_hit_positions.size() || r_hit_positions.is_empty());
  BLI_assert(ray_origins.size() == r_hit_normals.size() || r_hit_normals.is_empty());
  BLI_assert(ray_origins.size() == r_hit_distances.size() || r_hit_distances.is_empty());

  BVHTreeFromMesh tree_data;
  BKE_bvhtree_from_mesh_get(&tree_data, mesh, BVHTREE_FROM_LOOPTRI, 4);
  if (tree_data.tree == nullptr) {
    free_bvhtree_from_mesh(&tree_data);
    return;
  }

  for (const int i : ray_origins.index_range()) {
    const float ray_length = ray_lengths[i];
    const float3 ray_origin = ray_origins[i];
    const float3 ray_direction = ray_directions[i].normalized();

    BVHTreeRayHit hit;
    hit.index = -1;
    hit.dist = ray_length;
    if (BLI_bvhtree_ray_cast(tree_data.tree,
                             ray_origin,
                             ray_direction,
                             0.0f,
                             &hit,
                             tree_data.raycast_callback,
                             &tree_data) != -1) {
      if (!r_hit.is_empty()) {
        r_hit[i] = hit.index >= 0;
      }
      if (!r_hit_indices.is_empty()) {
        /* Index should always be a valid looptri index, use 0 when hit failed. */
        r_hit_indices[i] = max_ii(hit.index, 0);
      }
      if (!r_hit_positions.is_empty()) {
        r_hit_positions[i] = hit.co;
      }
      if (!r_hit_normals.is_empty()) {
        r_hit_normals[i] = hit.no;
      }
      if (!r_hit_distances.is_empty()) {
        r_hit_distances[i] = hit.dist;
      }
    }
    else {
      if (!r_hit.is_empty()) {
        r_hit[i] = false;
      }
      if (!r_hit_indices.is_empty()) {
        r_hit_indices[i] = 0;
      }
      if (!r_hit_positions.is_empty()) {
        r_hit_positions[i] = float3(0.0f, 0.0f, 0.0f);
      }
      if (!r_hit_normals.is_empty()) {
        r_hit_normals[i] = float3(0.0f, 0.0f, 0.0f);
      }
      if (!r_hit_distances.is_empty()) {
        r_hit_distances[i] = ray_length;
      }
    }
  }

  free_bvhtree_from_mesh(&tree_data);
}

static bke::mesh_surface_sample::eAttributeMapMode get_map_mode(
    GeometryNodeRaycastMapMode map_mode)
{
  switch (map_mode) {
    case GEO_NODE_RAYCAST_INTERPOLATED:
      return bke::mesh_surface_sample::eAttributeMapMode::INTERPOLATED;
    default:
    case GEO_NODE_RAYCAST_NEAREST:
      return bke::mesh_surface_sample::eAttributeMapMode::NEAREST;
  }
}

static void raycast_from_points(const GeoNodeExecParams &params,
                                const GeometrySet &target_geometry,
                                GeometryComponent &dst_component,
                                const AnonymousCustomDataLayerID *hit_id,
                                const AnonymousCustomDataLayerID *hit_position_id,
                                const AnonymousCustomDataLayerID *hit_normal_id,
                                const AnonymousCustomDataLayerID *hit_distance_id,
                                const Span<std::string> hit_attribute_names,
                                const Span<std::string> hit_attribute_output_names)
{
  BLI_assert(hit_attribute_names.size() == hit_attribute_output_names.size());

  const MeshComponent *src_mesh_component =
      target_geometry.get_component_for_read<MeshComponent>();
  if (src_mesh_component == nullptr) {
    return;
  }
  const Mesh *src_mesh = src_mesh_component->get_for_read();
  if (src_mesh == nullptr) {
    return;
  }
  if (src_mesh->totpoly == 0) {
    return;
  }

  const NodeGeometryRaycast &storage = *(const NodeGeometryRaycast *)params.node().storage;
  bke::mesh_surface_sample::eAttributeMapMode map_mode = get_map_mode(
      (GeometryNodeRaycastMapMode)storage.mapping);
  const AttributeDomain result_domain = ATTR_DOMAIN_POINT;

  GVArray_Typed<float3> ray_origins = dst_component.attribute_get_for_read<float3>(
      "position", result_domain, {0, 0, 0});

  bke::FieldRef<float3> direction_field = params.get_input_field<float3>("Ray Direction");
  bke::FieldInputs direction_field_inputs = direction_field->prepare_inputs();
  Vector<std::unique_ptr<bke::FieldInputValue>> direction_field_input_values;
  prepare_field_inputs(
      direction_field_inputs, dst_component, ATTR_DOMAIN_POINT, direction_field_input_values);
  bke::FieldOutput direction_field_output = direction_field->evaluate(
      IndexRange(ray_origins->size()), direction_field_inputs);
  GVArray_Typed<float3> ray_directions{direction_field_output.varray_ref()};

  bke::FieldRef<float> ray_length_field = params.get_input_field<float>("Ray Length");
  bke::FieldInputs ray_length_field_inputs = ray_length_field->prepare_inputs();
  Vector<std::unique_ptr<bke::FieldInputValue>> ray_length_field_input_values;
  prepare_field_inputs(
      ray_length_field_inputs, dst_component, ATTR_DOMAIN_POINT, ray_length_field_input_values);
  bke::FieldOutput ray_length_field_output = ray_length_field->evaluate(
      IndexRange(ray_origins->size()), ray_length_field_inputs);
  GVArray_Typed<float> ray_lengths{ray_length_field_output.varray_ref()};

  std::optional<OutputAttribute_Typed<bool>> is_hit_attribute;
  std::optional<OutputAttribute_Typed<float3>> hit_position_attribute;
  std::optional<OutputAttribute_Typed<float3>> hit_normal_attribute;
  std::optional<OutputAttribute_Typed<float>> hit_distance_attribute;

  if (hit_id != nullptr) {
    is_hit_attribute.emplace(dst_component.attribute_try_get_anonymous_for_output_only<bool>(
        *hit_id, ATTR_DOMAIN_POINT));
  }
  if (hit_position_id != nullptr) {
    hit_position_attribute.emplace(
        dst_component.attribute_try_get_anonymous_for_output_only<float3>(*hit_position_id,
                                                                          ATTR_DOMAIN_POINT));
  }
  if (hit_normal_id != nullptr) {
    hit_normal_attribute.emplace(dst_component.attribute_try_get_anonymous_for_output_only<float3>(
        *hit_normal_id, ATTR_DOMAIN_POINT));
  }
  if (hit_distance_id != nullptr) {
    hit_distance_attribute.emplace(
        dst_component.attribute_try_get_anonymous_for_output_only<float>(*hit_distance_id,
                                                                         ATTR_DOMAIN_POINT));
  }

  Array<int> hit_indices;
  if (!hit_attribute_names.is_empty()) {
    hit_indices.reinitialize(ray_origins->size());
  }

  MutableSpan<float3> hit_positions;
  Array<float3> hit_positions_internal;
  if (hit_position_attribute) {
    hit_positions = hit_position_attribute->as_span();
  }
  else {
    hit_positions_internal.reinitialize(ray_origins->size());
    hit_positions = hit_positions_internal;
  }

  raycast_to_mesh(src_mesh,
                  ray_origins,
                  ray_directions,
                  ray_lengths,
                  is_hit_attribute ? is_hit_attribute->as_span() : MutableSpan<bool>(),
                  hit_indices,
                  hit_positions,
                  hit_normal_attribute ? hit_normal_attribute->as_span() : MutableSpan<float3>(),
                  hit_distance_attribute ? hit_distance_attribute->as_span() :
                                           MutableSpan<float>());

  if (is_hit_attribute) {
    is_hit_attribute->save();
  }
  if (hit_position_attribute) {
    hit_position_attribute->save();
  }
  if (hit_normal_attribute) {
    hit_normal_attribute->save();
  }
  if (hit_distance_attribute) {
    hit_distance_attribute->save();
  }

  /* Custom interpolated attributes */
  bke::mesh_surface_sample::MeshAttributeInterpolator interp(src_mesh, hit_positions, hit_indices);
  for (const int i : hit_attribute_names.index_range()) {
    const std::optional<AttributeMetaData> meta_data = src_mesh_component->attribute_get_meta_data(
        hit_attribute_names[i]);
    if (meta_data) {
      ReadAttributeLookup hit_attribute = src_mesh_component->attribute_try_get_for_read(
          hit_attribute_names[i]);
      OutputAttribute hit_attribute_output = dst_component.attribute_try_get_for_output_only(
          hit_attribute_output_names[i], result_domain, meta_data->data_type);

      interp.sample_attribute(hit_attribute, hit_attribute_output, map_mode);

      hit_attribute_output.save();
    }
  }
}

static void geo_node_raycast_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet target_geometry_set = params.extract_input<GeometrySet>("Target Geometry");

  const Array<std::string> hit_attribute_names = {
      params.extract_input<std::string>("Target Attribute")};
  const Array<std::string> hit_attribute_output_names = {
      params.extract_input<std::string>("Hit Attribute")};

  geometry_set = bke::geometry_set_realize_instances(geometry_set);
  target_geometry_set = bke::geometry_set_realize_instances(target_geometry_set);

  AnonymousCustomDataLayerID *hit_id = nullptr;
  AnonymousCustomDataLayerID *hit_position_id = nullptr;
  AnonymousCustomDataLayerID *hit_normal_id = nullptr;
  AnonymousCustomDataLayerID *hit_distance_id = nullptr;
  if (params.output_is_required("Is Hit")) {
    hit_id = CustomData_anonymous_id_new("Is Hit");
  }
  if (params.output_is_required("Hit Position")) {
    hit_position_id = CustomData_anonymous_id_new("Hit Position");
  }
  if (params.output_is_required("Hit Normal")) {
    hit_normal_id = CustomData_anonymous_id_new("Hit Normal");
  }
  if (params.output_is_required("Hit Distance")) {
    hit_distance_id = CustomData_anonymous_id_new("Hit Distance");
  }

  static const Array<GeometryComponentType> types = {
      GEO_COMPONENT_TYPE_MESH, GEO_COMPONENT_TYPE_POINT_CLOUD, GEO_COMPONENT_TYPE_CURVE};
  for (const GeometryComponentType type : types) {
    if (geometry_set.has(type)) {
      raycast_from_points(params,
                          target_geometry_set,
                          geometry_set.get_component_for_write(type),
                          hit_id,
                          hit_position_id,
                          hit_normal_id,
                          hit_distance_id,
                          hit_attribute_names,
                          hit_attribute_output_names);
    }
  }

  if (hit_id != nullptr) {
    params.set_output(
        "Is Hit",
        bke::FieldRef<bool>(new bke::AnonymousAttributeField(*hit_id, CPPType::get<bool>())));
  }
  if (hit_position_id != nullptr) {
    params.set_output("Hit Position",
                      bke::FieldRef<float3>(new bke::AnonymousAttributeField(
                          *hit_position_id, CPPType::get<float3>())));
  }
  if (hit_normal_id != nullptr) {
    params.set_output("Hit Normal",
                      bke::FieldRef<float3>(new bke::AnonymousAttributeField(
                          *hit_normal_id, CPPType::get<float3>())));
  }
  if (hit_distance_id != nullptr) {
    params.set_output("Hit Distance",
                      bke::FieldRef<float>(new bke::AnonymousAttributeField(
                          *hit_distance_id, CPPType::get<float>())));
  }

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_raycast()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_RAYCAST, "Raycast", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_raycast_in, geo_node_raycast_out);
  node_type_size_preset(&ntype, NODE_SIZE_MIDDLE);
  node_type_init(&ntype, geo_node_raycast_init);
  node_type_storage(
      &ntype, "NodeGeometryRaycast", node_free_standard_storage, node_copy_standard_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_raycast_exec;
  ntype.draw_buttons = geo_node_raycast_layout;
  nodeRegisterType(&ntype);
}
