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
  /* The highest complexity data type for all attributes in the input meshes with the name. */
  CustomDataType data_type;
  /* The result domain is always "points" since we're creating a point cloud. */
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

BLI_NOINLINE static KDTree_3d *build_kdtree(Span<Vector<float3>> positions_array,
                                            const int initial_points_len)
{
  KDTree_3d *kdtree = BLI_kdtree_3d_new(initial_points_len);

  int i_point = 0;
  for (const int i_instance : positions_array.index_range()) {
    Span<float3> positions = positions_array[i_instance];
    for (const float3 position : positions) {
      BLI_kdtree_3d_insert(kdtree, i_point, position);
      i_point++;
    }
  }
  BLI_kdtree_3d_balance(kdtree);
  return kdtree;
}

BLI_NOINLINE static void update_elimination_mask_for_close_points(
    Span<Vector<float3>> positions_array,
    const float minimum_distance,
    MutableSpan<bool> elimination_mask,
    const int initial_points_len)
{
  if (minimum_distance <= 0.0f) {
    return;
  }

  KDTree_3d *kdtree = build_kdtree(positions_array, initial_points_len);

  /* The elimination mask is a flattened array for every point,
   * so keep track of the index to it separately. */
  int i_point = 0;
  for (Span<float3> positions : positions_array) {
    for (const float3 position : positions) {
      if (elimination_mask[i_point]) {
        i_point++;
        continue;
      }

      struct CallbackData {
        int index;
        MutableSpan<bool> elimination_mask;
      } callback_data = {i_point, elimination_mask};

      std::cout << "  KDTree nearest point callback: \n";
      BLI_kdtree_3d_range_search_cb(
          kdtree,
          position,
          minimum_distance,
          [](void *user_data, int index, const float *UNUSED(co), float UNUSED(dist_sq)) {
            CallbackData &callback_data = *static_cast<CallbackData *>(user_data);
            if (index != callback_data.index) {
              std::cout << "    Eliminating index mask: " << index << "\n";
              callback_data.elimination_mask[index] = true;
            }
            return true;
          },
          &callback_data);

      i_point++;
    }
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

template<typename T>
BLI_NOINLINE static void interpolate_attribute(const Mesh &mesh,
                                               Span<float3> bary_coords,
                                               Span<int> looptri_indices,
                                               const AttributeDomain source_domain,
                                               Span<T> source_span,
                                               MutableSpan<T> output_span)
{
  switch (source_domain) {
    case ATTR_DOMAIN_POINT: {
      interpolate_attribute_point<T>(mesh, bary_coords, looptri_indices, source_span, output_span);
      break;
    }
    case ATTR_DOMAIN_CORNER: {
      interpolate_attribute_corner<T>(
          mesh, bary_coords, looptri_indices, source_span, output_span);
      break;
    }
    default: {
      /* Not supported currently. */
      return;
    }
  }
}

BLI_NOINLINE static void interpolate_existing_attributes(
    Span<GeometryInstanceGroup> sets,
    Span<int> group_start_indices,
    Map<std::string, AttributeInfo> &attributes,
    GeometryComponent &component,
    Span<Vector<float3>> bary_coords_array,
    Span<Vector<int>> looptri_indices_array)
{
  for (blender::Map<std::string, AttributeInfo>::Item entry : attributes.items()) {
    StringRef attribute_name = entry.key;
    std::cout << "Working on attribute: " << attribute_name << "\n";

    const AttributeInfo attribute_info = entry.value;
    const CustomDataType output_data_type = attribute_info.data_type;
    OutputAttributePtr attribute_out = component.attribute_try_get_for_output(
        attribute_name, ATTR_DOMAIN_POINT, output_data_type);
    BLI_assert(attribute_out);
    if (!attribute_out) {
      continue;
    }

    attribute_math::convert_to_static_type(output_data_type, [&](auto dummy) {
      using T = decltype(dummy);

      MutableSpan<T> out_span = attribute_out->get_span_for_write_only<T>();

      int i_set_with_mesh = 0;
      int i_instance = 0;
      for (const GeometryInstanceGroup &set_group : sets) {
        const GeometrySet &set = set_group.geometry_set;
        std::cout << "  Working on geometry set: " << set << "\n";
        if (set.has_instances()) {
          std::cout << "    Set has instances\n";
        }
        if (set.has_pointcloud()) {
          std::cout << "    Set has point cloud\n";
        }
        if (set.has_volume()) {
          std::cout << "    Set has volume\n";
        }
        if (!set.has_mesh()) {
          std::cout << "    Set has no mesh\n";
          continue;
        }
        const MeshComponent &source_component = *set.get_component_for_read<MeshComponent>();
        const Mesh &mesh = *source_component.get_for_read();

        ReadAttributePtr dummy_attribute = source_component.attribute_try_get_for_read(
            attribute_name);
        if (!dummy_attribute) {
          std::cout << "    Source attribute not found\n";
          i_instance += set_group.transforms.size();
          i_set_with_mesh++;
          continue;
        }

        /* Do not interpolate the domain, that is handled by #interpolate_attribute. */
        const AttributeDomain source_domain = dummy_attribute->domain();

        ReadAttributePtr source_attribute = source_component.attribute_get_for_read(
            attribute_name, source_domain, output_data_type, nullptr);
        BLI_assert(source_attribute);
        Span<T> source_span = source_attribute->get_span<T>();

        if (!source_attribute) {
          std::cout << "    Source attribute read with correct domain not found\n";
          i_instance += set_group.transforms.size();
          i_set_with_mesh++;
          continue;
        }

        int i_point = group_start_indices[i_set_with_mesh];
        std::cout << "    Adding attribute from source, starting at " << i_point << "\n";
        for (const int UNUSED(i_set_instance) : set_group.transforms.index_range()) {
          Span<float3> bary_coords = bary_coords_array[i_instance].as_span();
          Span<int> looptri_indices = looptri_indices_array[i_instance].as_span();

          MutableSpan<T> instance_span = out_span.slice(i_point, bary_coords.size());
          interpolate_attribute<T>(
              mesh, bary_coords, looptri_indices, source_domain, source_span, instance_span);

          i_point += bary_coords.size();
          i_instance++;
        }
        i_set_with_mesh++;
      }
    });

    attribute_out.apply_span_and_save();
  }
}

BLI_NOINLINE static void compute_special_attributes(Span<GeometryInstanceGroup> sets,
                                                    GeometryComponent &component,
                                                    Span<Vector<float3>> bary_coords_array,
                                                    Span<Vector<int>> looptri_indices_array)
{
  OutputAttributePtr id_attribute = component.attribute_try_get_for_output(
      "id", ATTR_DOMAIN_POINT, CD_PROP_INT32);
  OutputAttributePtr normal_attribute = component.attribute_try_get_for_output(
      "normal", ATTR_DOMAIN_POINT, CD_PROP_FLOAT3);
  OutputAttributePtr rotation_attribute = component.attribute_try_get_for_output(
      "rotation", ATTR_DOMAIN_POINT, CD_PROP_FLOAT3);

  MutableSpan<int> ids_full = id_attribute->get_span_for_write_only<int>();
  MutableSpan<float3> normals_full = normal_attribute->get_span_for_write_only<float3>();
  MutableSpan<float3> rotations_full = rotation_attribute->get_span_for_write_only<float3>();

  int i_point = 0;
  int i_instance = 0;
  for (const GeometryInstanceGroup &set_group : sets) {
    const GeometrySet &set = set_group.geometry_set;
    if (!set.has_mesh()) {
      continue;
    }

    const MeshComponent &component = *set.get_component_for_read<MeshComponent>();
    const Mesh &mesh = *component.get_for_read();
    Span<MLoopTri> looptris = get_mesh_looptris(mesh);

    for (const int UNUSED(i_set_instance) : set_group.transforms.index_range()) {
      Span<float3> bary_coords = bary_coords_array[i_instance].as_span();
      Span<int> looptri_indices = looptri_indices_array[i_instance].as_span();
      MutableSpan<int> ids = ids_full.slice(i_point, bary_coords.size());
      MutableSpan<float3> normals = normals_full.slice(i_point, bary_coords.size());
      MutableSpan<float3> rotations = rotations_full.slice(i_point, bary_coords.size());

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

      i_instance++;
      i_point += bary_coords.size();
    }
  }

  id_attribute.apply_span_and_save();
  normal_attribute.apply_span_and_save();
  rotation_attribute.apply_span_and_save();
}

BLI_NOINLINE static void add_remaining_point_attributes(
    Span<GeometryInstanceGroup> sets,
    Span<int> group_start_indices,
    Map<std::string, AttributeInfo> &attributes,
    GeometryComponent &component,
    Span<Vector<float3>> bary_coords_array,
    Span<Vector<int>> looptri_indices_array)
{
  interpolate_existing_attributes(
      sets, group_start_indices, attributes, component, bary_coords_array, looptri_indices_array);
  compute_special_attributes(sets, component, bary_coords_array, looptri_indices_array);
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
      if (attribute_info.contains(name)) {
        AttributeInfo &info = attribute_info.lookup(name);
        info.data_type = attribute_data_type_highest_complexity({info.data_type, data_type});
      }
      else {
        attribute_info.add(name, {data_type});
      }
    }
  }
  return attribute_info;
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

  std::cout << "\nSCATTERING POINTS\n";

  Array<Vector<float3>> positions_array(instances_len);
  Array<Vector<float3>> bary_coords_array(instances_len);
  Array<Vector<int>> looptri_indices_array(instances_len);

  int initial_points_len = 0;
  int i_instance = 0;
  for (const GeometryInstanceGroup &set_group : sets) {
    const GeometrySet &set = set_group.geometry_set;
    if (!set.has_mesh()) {
      continue;
    }

    const MeshComponent &component = *set.get_component_for_read<MeshComponent>();
    const Mesh &mesh = *component.get_for_read();
    for (const float4x4 &transform : set_group.transforms) {
      Vector<float3> &positions = positions_array[i_instance];
      Vector<float3> &bary_coords = bary_coords_array[i_instance];
      Vector<int> &looptri_indices = looptri_indices_array[i_instance];

      switch (distribute_method) {
        case GEO_NODE_POINT_DISTRIBUTE_RANDOM: {
          const FloatReadAttribute density_factors = component.attribute_get_for_read<float>(
              density_attribute_name, ATTR_DOMAIN_POINT, 1.0f);
          initial_points_len += sample_mesh_surface(mesh,
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
          initial_points_len += sample_mesh_surface(
              mesh, transform, density, nullptr, seed, positions, bary_coords, looptri_indices);
          break;
      }
      i_instance++;
    }
  }

  std::cout << "  Scattered initial points: " << initial_points_len << "\n";

  if (distribute_method == GEO_NODE_POINT_DISTRIBUTE_POISSON) {
    /* Unlike the other result arrays, the elimination mask in stored as a flat array for every
     * point, in order to simplify culling points from the KDTree (which needs to know about all
     * points at once). */
    Array<bool> elimination_mask(initial_points_len, false);
    const float minimum_distance = params.get_input<float>("Distance Min");
    update_elimination_mask_for_close_points(
        positions_array, minimum_distance, elimination_mask, initial_points_len);

    int current_points_len = 0;
    for (const bool mask : elimination_mask) {
      if (!mask) {
        current_points_len++;
      }
    }
    std::cout << "  Eliminated based on KDTree, elimination mask total: " << current_points_len
              << "\n";

    int i_point = 0;
    i_instance = 0;
    for (const GeometryInstanceGroup &set_group : sets) {
      const GeometrySet &set = set_group.geometry_set;
      if (!set.has_mesh()) {
        continue;
      }

      const MeshComponent &component = *set.get_component_for_read<MeshComponent>();
      const Mesh &mesh = *component.get_for_read();
      const FloatReadAttribute density_factors = component.attribute_get_for_read<float>(
          density_attribute_name, ATTR_DOMAIN_POINT, 1.0f);

      for (const int UNUSED(i_set_instance) : set_group.transforms.index_range()) {
        Vector<float3> &positions = positions_array[i_instance];
        Vector<float3> &bary_coords = bary_coords_array[i_instance];
        Vector<int> &looptri_indices = looptri_indices_array[i_instance];

        update_elimination_mask_based_on_density_factors(
            mesh,
            density_factors,
            bary_coords,
            looptri_indices,
            elimination_mask.as_mutable_span().slice(i_point, positions.size()));

        /* The positions vector's size is changed, temporarily store the
         * original size to properly advance the elimination mask index. */
        const int initial_positions_size = positions.size();
        eliminate_points_based_on_mask(elimination_mask.as_span().slice(i_point, positions.size()),
                                       positions,
                                       bary_coords,
                                       looptri_indices);

        i_point += initial_positions_size;
        i_instance++;
      }
    }

    current_points_len = 0;
    for (const bool mask : elimination_mask) {
      if (!mask) {
        current_points_len++;
      }
    }
    std::cout << "  Eliminated based on density, elimination mask total: " << current_points_len
              << "\n";
  }

  int final_points_len = 0;
  Array<int> group_start_indices(sets.size());
  for (const int i : positions_array.index_range()) {
    Vector<float3> &positions = positions_array[i];
    group_start_indices[i] = final_points_len;
    final_points_len += positions.size();
  }

  std::cout << "  Elinimated points, now there are: " << final_points_len << "\n";

  PointCloud *pointcloud = BKE_pointcloud_new_nomain(final_points_len);
  int i_point = 0;
  for (Vector<float3> &positions : positions_array) {
    memcpy(pointcloud->co + i_point, positions.data(), sizeof(float3) * positions.size());
    i_point += positions.size();
  }

  MutableSpan(pointcloud->radius, pointcloud->totpoint).fill(0.05f);

  PointCloudComponent &point_component =
      geometry_set_out.get_component_for_write<PointCloudComponent>();
  point_component.replace(pointcloud);

  std::cout << "\nINTERPOLATING ATTRIBUTES\n";

  Map<std::string, AttributeInfo> attributes = gather_attribute_info(sets);
  add_remaining_point_attributes(sets,
                                 group_start_indices,
                                 attributes,
                                 point_component,
                                 bary_coords_array,
                                 looptri_indices_array);

  std::cout << "Final geomtry set: " << geometry_set_out << "\n";

  Set<std::string> final_attribute_names = point_component.attribute_names();
  std::cout << "Final attribute names\n";
  for (std::string name : final_attribute_names) {
    std::cout << "  " << name << "\n";
  }
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
