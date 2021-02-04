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

#include "BLI_float3.hh"
#include "BLI_hash.h"
#include "BLI_kdtree.h"
#include "BLI_math_vector.h"
#include "BLI_rand.hh"
#include "BLI_span.hh"
#include "BLI_timeit.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_pointcloud_types.h"

#include "BKE_attribute_math.hh"
#include "BKE_bvhutils.h"
#include "BKE_deform.h"
#include "BKE_mesh.h"
#include "BKE_mesh_runtime.h"
#include "BKE_pointcloud.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_point_distribute_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Distance Min"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100000.0f, PROP_NONE},
    {SOCK_FLOAT, N_("Density Max"), 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 100000.0f, PROP_NONE},
    {SOCK_STRING, N_("Density Attribute")},
    {SOCK_INT, N_("Seed"), 0, 0, 0, 0, -10000, 10000},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_point_distribute_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void node_point_distribute_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  bNodeSocket *sock_min_dist = (bNodeSocket *)BLI_findlink(&node->inputs, 1);

  nodeSetSocketAvailability(sock_min_dist, ELEM(node->custom1, GEO_NODE_POINT_DISTRIBUTE_POISSON));
}

namespace blender::nodes {

struct AttributeInfo {
  CustomDataType data_type;
  AttributeDomain domain;
};

/**
 * Use an arbitrary choice of axes for a usable rotation attribute directly out of this node.
 */
static float3 normal_to_euler_rotation(const float3 normal)
{
  float quat[4];
  vec_to_quat(quat, normal, OB_NEGZ, OB_POSY);
  float3 rotation;
  quat_to_eul(rotation, quat);
  return rotation;
}

static Span<MLoopTri> get_mesh_looptris(const Mesh &mesh)
{
  /* This only updates a cache and can be considered to be logically const. */
  const MLoopTri *looptris = BKE_mesh_runtime_looptri_ensure(const_cast<Mesh *>(&mesh));
  const int looptris_len = BKE_mesh_runtime_looptri_len(&mesh);
  return {looptris, looptris_len};
}

static int sample_mesh_surface(const Mesh &mesh,
                               const float4x4 &transform,
                               const float base_density,
                               const FloatReadAttribute *density_factors,
                               const int seed,
                               Vector<float3> &r_positions,
                               Vector<float3> &r_bary_coords,
                               Vector<int> &r_looptri_indices)
{
  Span<MLoopTri> looptris = get_mesh_looptris(mesh);

  int points_len = 0;
  for (const int looptri_index : looptris.index_range()) {
    const MLoopTri &looptri = looptris[looptri_index];
    const int v0_index = mesh.mloop[looptri.tri[0]].v;
    const int v1_index = mesh.mloop[looptri.tri[1]].v;
    const int v2_index = mesh.mloop[looptri.tri[2]].v;

    const float3 v0_pos = transform * float3(mesh.mvert[v0_index].co);
    const float3 v1_pos = transform * float3(mesh.mvert[v1_index].co);
    const float3 v2_pos = transform * float3(mesh.mvert[v2_index].co);

    float looptri_density_factor = 1.0f;
    if (density_factors != nullptr) {
      const float v0_density_factor = std::max(0.0f, (*density_factors)[v0_index]);
      const float v1_density_factor = std::max(0.0f, (*density_factors)[v1_index]);
      const float v2_density_factor = std::max(0.0f, (*density_factors)[v2_index]);
      looptri_density_factor = (v0_density_factor + v1_density_factor + v2_density_factor) / 3.0f;
    }
    const float area = area_tri_v3(v0_pos, v1_pos, v2_pos);

    const int looptri_seed = BLI_hash_int(looptri_index + seed);
    RandomNumberGenerator looptri_rng(looptri_seed);

    const float points_amount_fl = area * base_density * looptri_density_factor;
    const float add_point_probability = fractf(points_amount_fl);
    const bool add_point = add_point_probability > looptri_rng.get_float();
    const int point_amount = (int)points_amount_fl + (int)add_point;

    for (int i = 0; i < point_amount; i++) {
      const float3 bary_coord = looptri_rng.get_barycentric_coordinates();
      float3 point_pos;
      interp_v3_v3v3v3(point_pos, v0_pos, v1_pos, v2_pos, bary_coord);
      r_positions.append(point_pos);
      r_bary_coords.append(bary_coord);
      r_looptri_indices.append(looptri_index);
      points_len++;
    }
  }
  return points_len;
}

BLI_NOINLINE static KDTree_3d *build_kdtree(Span<float3> positions)
{
  KDTree_3d *kdtree = BLI_kdtree_3d_new(positions.size());
  for (const int i : positions.index_range()) {
    BLI_kdtree_3d_insert(kdtree, i, positions[i]);
  }
  BLI_kdtree_3d_balance(kdtree);
  return kdtree;
}

BLI_NOINLINE static void update_elimination_mask_for_close_points(
    Span<float3> positions, const float minimum_distance, MutableSpan<bool> elimination_mask)
{
  if (minimum_distance <= 0.0f) {
    return;
  }

  KDTree_3d *kdtree = build_kdtree(positions);

  for (const int i : positions.index_range()) {
    if (elimination_mask[i]) {
      continue;
    }

    struct CallbackData {
      int index;
      MutableSpan<bool> elimination_mask;
    } callback_data = {i, elimination_mask};

    BLI_kdtree_3d_range_search_cb(
        kdtree,
        positions[i],
        minimum_distance,
        [](void *user_data, int index, const float *UNUSED(co), float UNUSED(dist_sq)) {
          CallbackData &callback_data = *static_cast<CallbackData *>(user_data);
          if (index != callback_data.index) {
            callback_data.elimination_mask[index] = true;
          }
          return true;
        },
        &callback_data);
  }
  BLI_kdtree_3d_free(kdtree);
}

BLI_NOINLINE static void update_elimination_mask_based_on_density_factors(
    const Mesh &mesh,
    const FloatReadAttribute &density_factors,
    Span<float3> bary_coords,
    Span<int> looptri_indices,
    MutableSpan<bool> elimination_mask)
{
  Span<MLoopTri> looptris = get_mesh_looptris(mesh);
  for (const int i : bary_coords.index_range()) {
    if (elimination_mask[i]) {
      continue;
    }

    const MLoopTri &looptri = looptris[looptri_indices[i]];
    const float3 bary_coord = bary_coords[i];

    const int v0_index = mesh.mloop[looptri.tri[0]].v;
    const int v1_index = mesh.mloop[looptri.tri[1]].v;
    const int v2_index = mesh.mloop[looptri.tri[2]].v;

    const float v0_density_factor = std::max(0.0f, density_factors[v0_index]);
    const float v1_density_factor = std::max(0.0f, density_factors[v1_index]);
    const float v2_density_factor = std::max(0.0f, density_factors[v2_index]);

    const float probablity = attribute_math::mix3<float>(
        bary_coord, v0_density_factor, v1_density_factor, v2_density_factor);

    const float hash = BLI_hash_int_01(bary_coord.hash());
    if (hash > probablity) {
      elimination_mask[i] = true;
    }
  }
}

BLI_NOINLINE static void eliminate_points_based_on_mask(Span<bool> elimination_mask,
                                                        Vector<float3> &positions,
                                                        Vector<float3> &bary_coords,
                                                        Vector<int> &looptri_indices)
{
  for (int i = positions.size() - 1; i >= 0; i--) {
    if (elimination_mask[i]) {
      positions.remove_and_reorder(i);
      bary_coords.remove_and_reorder(i);
      looptri_indices.remove_and_reorder(i);
    }
  }
}

template<typename T>
BLI_NOINLINE static void interpolate_attribute_point(const Mesh &mesh,
                                                     const Span<float3> bary_coords,
                                                     const Span<int> looptri_indices,
                                                     const Span<T> data_in,
                                                     MutableSpan<T> data_out)
{
  BLI_assert(data_in.size() == mesh.totvert);
  Span<MLoopTri> looptris = get_mesh_looptris(mesh);

  for (const int i : bary_coords.index_range()) {
    const int looptri_index = looptri_indices[i];
    const MLoopTri &looptri = looptris[looptri_index];
    const float3 &bary_coord = bary_coords[i];

    const int v0_index = mesh.mloop[looptri.tri[0]].v;
    const int v1_index = mesh.mloop[looptri.tri[1]].v;
    const int v2_index = mesh.mloop[looptri.tri[2]].v;

    const T &v0 = data_in[v0_index];
    const T &v1 = data_in[v1_index];
    const T &v2 = data_in[v2_index];

    const T interpolated_value = attribute_math::mix3(bary_coord, v0, v1, v2);
    data_out[i] = interpolated_value;
  }
}

template<typename T>
BLI_NOINLINE static void interpolate_attribute_corner(const Mesh &mesh,
                                                      const Span<float3> bary_coords,
                                                      const Span<int> looptri_indices,
                                                      const Span<T> data_in,
                                                      MutableSpan<T> data_out)
{
  BLI_assert(data_in.size() == mesh.totloop);
  Span<MLoopTri> looptris = get_mesh_looptris(mesh);

  for (const int i : bary_coords.index_range()) {
    const int looptri_index = looptri_indices[i];
    const MLoopTri &looptri = looptris[looptri_index];
    const float3 &bary_coord = bary_coords[i];

    const int loop_index_0 = looptri.tri[0];
    const int loop_index_1 = looptri.tri[1];
    const int loop_index_2 = looptri.tri[2];

    const T &v0 = data_in[loop_index_0];
    const T &v1 = data_in[loop_index_1];
    const T &v2 = data_in[loop_index_2];

    const T interpolated_value = attribute_math::mix3(bary_coord, v0, v1, v2);
    data_out[i] = interpolated_value;
  }
}

BLI_NOINLINE static void interpolate_attribute(const Mesh &mesh,
                                               Span<float3> bary_coords,
                                               Span<int> looptri_indices,
                                               const StringRef attribute_name,
                                               const ReadAttribute &attribute_in,
                                               GeometryComponent &component)
{
  const CustomDataType data_type = attribute_in.custom_data_type();
  const AttributeDomain domain = attribute_in.domain();
  if (!ELEM(domain, ATTR_DOMAIN_POINT, ATTR_DOMAIN_CORNER)) {
    /* Not supported currently. */
    return;
  }

  OutputAttributePtr attribute_out = component.attribute_try_get_for_output(
      attribute_name, ATTR_DOMAIN_POINT, data_type);
  if (!attribute_out) {
    return;
  }

  attribute_math::convert_to_static_type(data_type, [&](auto dummy) {
    using T = decltype(dummy);

    Span data_in = attribute_in.get_span<T>();
    MutableSpan data_out = attribute_out->get_span_for_write_only<T>();

    switch (domain) {
      case ATTR_DOMAIN_POINT: {
        interpolate_attribute_point<T>(mesh, bary_coords, looptri_indices, data_in, data_out);
        break;
      }
      case ATTR_DOMAIN_CORNER: {
        interpolate_attribute_corner<T>(mesh, bary_coords, looptri_indices, data_in, data_out);
        break;
      }
      default: {
        BLI_assert(false);
        break;
      }
    }
  });
  attribute_out.apply_span_and_save();
}

BLI_NOINLINE static void interpolate_existing_attributes(const MeshComponent &mesh_component,
                                                         GeometryComponent &component,
                                                         Span<float3> bary_coords,
                                                         Span<int> looptri_indices)
{
  const Mesh &mesh = *mesh_component.get_for_read();

  Set<std::string> attribute_names = mesh_component.attribute_names();
  for (StringRefNull attribute_name : attribute_names) {
    if (ELEM(attribute_name, "position", "normal", "id")) {
      continue;
    }

    ReadAttributePtr attribute_in = mesh_component.attribute_try_get_for_read(attribute_name);
    interpolate_attribute(
        mesh, bary_coords, looptri_indices, attribute_name, *attribute_in, component);
  }
}

BLI_NOINLINE static void compute_special_attributes(const Mesh &mesh,
                                                    GeometryComponent &component,
                                                    Span<float3> bary_coords,
                                                    Span<int> looptri_indices)
{
  OutputAttributePtr id_attribute = component.attribute_try_get_for_output(
      "id", ATTR_DOMAIN_POINT, CD_PROP_INT32);
  OutputAttributePtr normal_attribute = component.attribute_try_get_for_output(
      "normal", ATTR_DOMAIN_POINT, CD_PROP_FLOAT3);
  OutputAttributePtr rotation_attribute = component.attribute_try_get_for_output(
      "rotation", ATTR_DOMAIN_POINT, CD_PROP_FLOAT3);

  MutableSpan<int> ids = id_attribute->get_span_for_write_only<int>();
  MutableSpan<float3> normals = normal_attribute->get_span_for_write_only<float3>();
  MutableSpan<float3> rotations = rotation_attribute->get_span_for_write_only<float3>();

  Span<MLoopTri> looptris = get_mesh_looptris(mesh);
  for (const int i : bary_coords.index_range()) {
    const int looptri_index = looptri_indices[i];
    const MLoopTri &looptri = looptris[looptri_index];
    const float3 &bary_coord = bary_coords[i];

    const int v0_index = mesh.mloop[looptri.tri[0]].v;
    const int v1_index = mesh.mloop[looptri.tri[1]].v;
    const int v2_index = mesh.mloop[looptri.tri[2]].v;
    const float3 v0_pos = mesh.mvert[v0_index].co;
    const float3 v1_pos = mesh.mvert[v1_index].co;
    const float3 v2_pos = mesh.mvert[v2_index].co;

    ids[i] = (int)(bary_coord.hash()) + looptri_index;
    normal_tri_v3(normals[i], v0_pos, v1_pos, v2_pos);
    rotations[i] = normal_to_euler_rotation(normals[i]);
  }

  id_attribute.apply_span_and_save();
  normal_attribute.apply_span_and_save();
  rotation_attribute.apply_span_and_save();
}

BLI_NOINLINE static void add_remaining_point_attributes(Span<GeometryInstanceGroup> sets,
                                                        GeometryComponent &component,
                                                        Span<float3> bary_coords,
                                                        Span<int> looptri_indices)
{
  /* TODO: This needs some more thought. The problem is that we need to know which instance /
   * component the data came from. A map from #i_instance to #i_point will probably be necessary
   * to support fast coping of attributes. */
  // interpolate_existing_attributes(mesh_component, component, bary_coords, looptri_indices);
  // compute_special_attributes(
  //     *mesh_component.get_for_read(), component, bary_coords, looptri_indices);
}

static Map<std::string, AttributeInfo> gather_attribute_info(
    Span<GeometryInstanceGroup> geometry_sets)
{
  Map<std::string, AttributeInfo> attribute_info;
  for (const GeometryInstanceGroup &set_group : geometry_sets) {
    const GeometrySet &set = set_group.geometry_set;
    if (!set.has_mesh()) {
      continue;
    }
    const MeshComponent &component = *set.get_component_for_read<MeshComponent>();

    for (const std::string &name : component.attribute_names()) {
      if (ELEM(name, "position", "normal", "id")) {
        continue;
      }
      ReadAttributePtr attribute = component.attribute_try_get_for_read(name);
      BLI_assert(attribute);
      const CustomDataType data_type = attribute->custom_data_type();
      const AttributeDomain domain = attribute->domain();
      if (attribute_info.contains(name)) {
        AttributeInfo &info = attribute_info.lookup(name);
        info.data_type = attribute_data_type_highest_complexity({info.data_type, data_type});
        /* TODO: Choose the domain based on priority. */
        info.domain = domain;
      }
      else {
        attribute_info.add(name, {data_type, domain});
      }
    }
  }
  return attribute_info;
}

static void calculate_instance_point_lengths_after_elimination(
    MutableSpan<int> instance_point_lengths, Span<bool> elimination_mask)
{
  int i_point_offset = 0;
  for (const int i_instance : instance_point_lengths.index_range()) {
    Span<bool> instance_elimination_mask = elimination_mask.slice(
        i_point_offset, instance_point_lengths[i_instance]);
    int final_length = 0;
    for (const int UNUSED(i) : instance_elimination_mask) {
      final_length++;
    }
    instance_point_lengths[i_instance] = final_length;
  }
}

static void geo_node_point_distribute_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet geometry_set_out;

  const GeometryNodePointDistributeMethod distribute_method =
      static_cast<GeometryNodePointDistributeMethod>(params.node().custom1);

  if (!geometry_set.has_mesh() && !geometry_set.has_instances()) {
    params.set_output("Geometry", std::move(geometry_set_out));
    return;
  }

  const float density = params.extract_input<float>("Density Max");
  const std::string density_attribute_name = params.extract_input<std::string>(
      "Density Attribute");

  if (density <= 0.0f) {
    params.set_output("Geometry", std::move(geometry_set_out));
    return;
  }

  const int seed = params.get_input<int>("Seed");

  Vector<GeometryInstanceGroup> sets = BKE_geometry_set_gather_instanced(geometry_set);
  int instances_len = 0;
  for (GeometryInstanceGroup set_group : sets) {
    instances_len += set_group.transforms.size();
  }

  Array<int> instance_point_lengths(instances_len);
  Vector<float3> bary_coords;
  Vector<int> looptri_indices;
  Vector<float3> positions;

  int i_instance = 0;
  for (const int i_set : sets.index_range()) {
    const GeometryInstanceGroup &set_group = sets[i_set];
    const GeometrySet &set = set_group.geometry_set;
    if (!set.has_mesh()) {
      continue;
    }

    const MeshComponent &component = *set.get_component_for_read<MeshComponent>();
    const Mesh &mesh = *component.get_for_read();
    for (const float4x4 &transform : set_group.transforms) {
      int points_len;
      switch (distribute_method) {
        case GEO_NODE_POINT_DISTRIBUTE_RANDOM: {
          const FloatReadAttribute density_factors = component.attribute_get_for_read<float>(
              density_attribute_name, ATTR_DOMAIN_POINT, 1.0f);
          points_len = sample_mesh_surface(mesh,
                                           transform,
                                           density,
                                           &density_factors,
                                           seed,
                                           positions,
                                           bary_coords,
                                           looptri_indices);
          break;
        }
        case GEO_NODE_POINT_DISTRIBUTE_POISSON:
          points_len = sample_mesh_surface(
              mesh, transform, density, nullptr, seed, positions, bary_coords, looptri_indices);
          break;
      }
      instance_point_lengths[i_instance] = points_len;
      i_instance++;
    }
  }

  if (distribute_method == GEO_NODE_POINT_DISTRIBUTE_POISSON) {
    Array<bool> elimination_mask(positions.size(), false);
    const float minimum_distance = params.get_input<float>("Distance Min");
    update_elimination_mask_for_close_points(positions, minimum_distance, elimination_mask);

    i_instance = 0;
    int i_point = 0;
    for (const int i_set : sets.index_range()) {
      const GeometryInstanceGroup &set_group = sets[i_set];
      const GeometrySet &set = set_group.geometry_set;
      if (!set.has_mesh()) {
        continue;
      }

      const MeshComponent &component = *set.get_component_for_read<MeshComponent>();
      const Mesh &mesh = *component.get_for_read();

      for (const int UNUSED(i_set_instance) : set_group.transforms.index_range()) {
        const FloatReadAttribute density_factors = component.attribute_get_for_read<float>(
            density_attribute_name, ATTR_DOMAIN_POINT, 1.0f);

        update_elimination_mask_based_on_density_factors(
            mesh,
            density_factors,
            bary_coords.as_span().slice(i_point, instance_point_lengths[i_instance]),
            looptri_indices.as_span().slice(i_point, instance_point_lengths[i_instance]),
            elimination_mask.as_mutable_span().slice(i_point, instance_point_lengths[i_instance]));

        i_point += instance_point_lengths[i_instance];
        i_instance++;
      }
    }

    eliminate_points_based_on_mask(elimination_mask, positions, bary_coords, looptri_indices);
  }

  PointCloud *pointcloud = BKE_pointcloud_new_nomain(positions.size());
  memcpy(pointcloud->co, positions.data(), sizeof(float3) * positions.size());
  MutableSpan(pointcloud->radius, pointcloud->totpoint).fill(0.05f);
  for (const int i : IndexRange(pointcloud->totpoint)) {
    pointcloud->radius[i] = 0.05f;
  }

  PointCloudComponent &point_component =
      geometry_set_out.get_component_for_write<PointCloudComponent>();
  point_component.replace(pointcloud);

  Map<std::string, AttributeInfo> attributes = gather_attribute_info(sets);
  add_remaining_point_attributes(sets, point_component, bary_coords, looptri_indices);

  params.set_output("Geometry", std::move(geometry_set_out));
}

}  // namespace blender::nodes

void register_node_type_geo_point_distribute()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_POINT_DISTRIBUTE, "Point Distribute", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_point_distribute_in, geo_node_point_distribute_out);
  node_type_update(&ntype, node_point_distribute_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_point_distribute_exec;
  nodeRegisterType(&ntype);
}
