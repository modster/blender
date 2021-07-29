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

#include "BLI_kdopbvh.h"
#include "BLI_math_base_safe.h"
#include "BLI_task.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_pointcloud_types.h"

#include "BKE_attribute_access.hh"
#include "BKE_attribute_math.hh"
#include "BKE_bvhutils.h"
#include "BKE_colortools.h"
#include "BKE_mesh_runtime.h"
#include "BKE_mesh_sample.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_attribute_range_query_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_GEOMETRY, N_("Source Geometry")},
    {SOCK_STRING, N_("Radius")},
    {SOCK_FLOAT, N_("Radius"), 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, FLT_MAX},
    {SOCK_STRING, N_("Source")},
    {SOCK_STRING, N_("Destination")},
    {SOCK_STRING, N_("Count")},
    {SOCK_STRING, N_("Total Weight")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_attribute_range_query_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

static void geo_node_attribute_range_query_layout(uiLayout *layout,
                                                  bContext *UNUSED(C),
                                                  PointerRNA *ptr)
{
  const bNode *node = (const bNode *)ptr->data;
  const NodeGeometryAttributeRangeQuery &node_storage = *(NodeGeometryAttributeRangeQuery *)
                                                             node->storage;

  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "domain", 0, IFACE_("Domain"), ICON_NONE);
  uiItemR(layout, ptr, "mode", 0, IFACE_("Mode"), ICON_NONE);

  uiItemR(layout, ptr, "input_type_radius", 0, IFACE_("Radius"), ICON_NONE);

  if (node_storage.mode == GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF) {
    uiItemR(layout, ptr, "falloff_type", 0, IFACE_("Invert Type"), ICON_NONE);
    uiItemR(layout, ptr, "invert_falloff", 0, IFACE_("Invert Falloff"), ICON_NONE);

    if (node_storage.falloff_type == GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF_CURVE) {
      uiTemplateCurveMapping(layout, ptr, "falloff_curve", 0, false, false, false, false);
    }
  }
}

static void geo_node_attribute_range_query_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryAttributeRangeQuery *data = (NodeGeometryAttributeRangeQuery *)MEM_callocN(
      sizeof(NodeGeometryAttributeRangeQuery), __func__);
  data->domain = ATTR_DOMAIN_AUTO;
  data->mode = GEO_NODE_ATTRIBUTE_RANGE_QUERY_CLOSEST;
  data->falloff_type = GEO_NODE_ATTRIBUTE_RANGE_QUERY_AVERAGE;
  data->falloff_curve = BKE_curvemapping_add(1, 0.0f, 0.0f, 1.0f, 1.0f);
  data->input_type_radius = GEO_NODE_ATTRIBUTE_INPUT_FLOAT;
  node->storage = data;
}

static void geo_node_attribute_range_query_free_storage(bNode *node)
{
  if (node->storage) {
    NodeGeometryAttributeRangeQuery *data = (NodeGeometryAttributeRangeQuery *)node->storage;
    BKE_curvemapping_free(data->falloff_curve);
    MEM_freeN(node->storage);
  }
}

static void geo_node_attribute_range_query_copy_storage(bNodeTree *UNUSED(dest_ntree),
                                                        bNode *dest_node,
                                                        const bNode *src_node)
{
  dest_node->storage = MEM_dupallocN(src_node->storage);
  NodeGeometryAttributeRangeQuery *src_data = (NodeGeometryAttributeRangeQuery *)src_node->storage;
  NodeGeometryAttributeRangeQuery *dest_data = (NodeGeometryAttributeRangeQuery *)
                                                   dest_node->storage;
  dest_data->falloff_curve = BKE_curvemapping_copy(src_data->falloff_curve);
}

static void geo_node_attribute_range_query_update(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryAttributeRangeQuery &node_storage = *(NodeGeometryAttributeRangeQuery *)
                                                       node->storage;

  blender::nodes::update_attribute_input_socket_availabilities(
      *node, "Radius", (GeometryNodeAttributeInputMode)node_storage.input_type_radius);
}

namespace blender::nodes {

struct TypeDetails {
  static void add(float &sum, float value)
  {
    sum += value;
  }

  static void add(float2 &sum, float2 value)
  {
    sum += value;
  }

  static void add(float3 &sum, float3 value)
  {
    sum += value;
  }

  static void add(int &sum, int value)
  {
    sum += value;
  }

  static void add(bool &sum, bool value)
  {
    sum |= value;
  }

  static void add(ColorGeometry4f &sum, ColorGeometry4f value)
  {
    sum.r += value.r;
    sum.g += value.g;
    sum.b += value.b;
    sum.a += value.a;
  }

  static void add_weighted(float &sum, float value, float weight)
  {
    sum += value * weight;
  }

  static void add_weighted(float2 &sum, float2 value, float weight)
  {
    sum += value * weight;
  }

  static void add_weighted(float3 &sum, float3 value, float weight)
  {
    sum += value * weight;
  }

  static void add_weighted(int &sum, int value, float weight)
  {
    sum += (int)((float)value * weight);
  }

  static void add_weighted(bool &sum, bool value, float weight)
  {
    if (weight > 0.0f) {
      sum |= value;
    }
  }

  static void add_weighted(ColorGeometry4f &sum, ColorGeometry4f value, float weight)
  {
    sum.r += value.r * weight;
    sum.g += value.g * weight;
    sum.b += value.b * weight;
    sum.a += value.a * weight;
  }

  static float normalize(float sum, float total_weight)
  {
    return safe_divide(sum, total_weight);
  }

  static float2 normalize(float2 sum, float total_weight)
  {
    return float2(safe_divide(sum[0], total_weight), safe_divide(sum[1], total_weight));
  }

  static float3 normalize(float3 sum, float total_weight)
  {
    return float3::safe_divide(sum, float3(total_weight));
  }

  static int normalize(int sum, float total_weight)
  {
    return total_weight != 0.0f ? (int)((float)sum / total_weight) : 0;
  }

  static bool normalize(bool sum, float total_weight)
  {
    return sum ? (total_weight > 0.0f) : false;
  }

  static ColorGeometry4f normalize(ColorGeometry4f sum, float total_weight)
  {
    return ColorGeometry4f(safe_divide(sum.r, total_weight),
                           safe_divide(sum.g, total_weight),
                           safe_divide(sum.b, total_weight),
                           safe_divide(sum.a, total_weight));
  }

  static float min(float a, float b)
  {
    return min_ff(a, b);
  }

  static float2 min(float2 a, float2 b)
  {
    return float2(min_ff(a.x, b.x), min_ff(a.y, b.y));
  }

  static float3 min(float3 a, float3 b)
  {
    return float3(min_ff(a.x, b.x), min_ff(a.y, b.y), min_ff(a.z, b.z));
  }

  static int min(int a, int b)
  {
    return min_ii(a, b);
  }

  static bool min(bool a, bool b)
  {
    return a && b;
  }

  static ColorGeometry4f min(ColorGeometry4f a, ColorGeometry4f b)
  {
    return ColorGeometry4f(min_ff(a.r, b.r), min_ff(a.g, b.g), min_ff(a.b, b.b), min_ff(a.a, b.a));
  }

  static float max(float a, float b)
  {
    return max_ff(a, b);
  }

  static float2 max(float2 a, float2 b)
  {
    return float2(max_ff(a.x, b.x), max_ff(a.y, b.y));
  }

  static float3 max(float3 a, float3 b)
  {
    return float3(max_ff(a.x, b.x), max_ff(a.y, b.y), max_ff(a.z, b.z));
  }

  static int max(int a, int b)
  {
    return max_ii(a, b);
  }

  static bool max(bool a, bool b)
  {
    return a || b;
  }

  static ColorGeometry4f max(ColorGeometry4f a, ColorGeometry4f b)
  {
    return ColorGeometry4f(max_ff(a.r, b.r), max_ff(a.g, b.g), max_ff(a.b, b.b), max_ff(a.a, b.a));
  }
};

template<typename ValueType> struct RangeQueryData {
  ValueType result_;
  float total_weight_;

  /* For relative distance in falloff mode. */
  float radius_;
  /* For closest-point mode. */
  float min_dist_sq_;
};

template<typename ValueType, typename AccumulatorType> struct RangeQueryUserData {
  const AccumulatorType accum_;
  const GVArray_Typed<ValueType> *values_;

  RangeQueryData<ValueType> data_;

  RangeQueryUserData(const AccumulatorType &accum,
                     const GVArray_Typed<ValueType> &values,
                     float radius)
      : accum_(accum), values_(&values)
  {
    BLI_assert(radius > 0.0f);
    data_.radius_ = radius;
    data_.result_ = ValueType(0);
    data_.total_weight_ = 0.0f;
    data_.min_dist_sq_ = FLT_MAX;
  }

  static void callback(void *userdata, int index, const float co[3], float dist_sq)
  {
    RangeQueryUserData<ValueType, AccumulatorType> &calldata = *(
        RangeQueryUserData<ValueType, AccumulatorType> *)userdata;

    ValueType value = (*calldata.values_)[index];
    calldata.accum_.add_point(calldata.data_, float3(co), dist_sq, value);
  }
};

struct RangeQueryAccumulator_Average {
  template<typename ValueType>
  void add_point(RangeQueryData<ValueType> &data,
                 const float3 &co,
                 float dist_sq,
                 ValueType value) const
  {
    TypeDetails::add(data.result_, value);
    data.total_weight_ += 1.0f;
  };
};

struct RangeQueryAccumulator_Sum {
  template<typename ValueType>
  void add_point(RangeQueryData<ValueType> &data,
                 const float3 &co,
                 float dist_sq,
                 ValueType value) const
  {
    TypeDetails::add(data.result_, value);
    /* Set weight to 1 so normalization leaves the sum unchanged. */
    data.total_weight_ = 1.0f;
  };
};

struct RangeQueryAccumulator_Falloff {
  GeometryNodeAttributeRangeQueryFalloffType falloff_type_;
  bool invert_;
  CurveMapping *curve_map_;

  RangeQueryAccumulator_Falloff(GeometryNodeAttributeRangeQueryFalloffType falloff_type,
                                bool invert,
                                CurveMapping *curve_map)
      : falloff_type_(falloff_type), invert_(invert), curve_map_(curve_map)
  {
    if (falloff_type == GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF_CURVE) {
      BKE_curvemapping_init(curve_map_);
    }
  }

  float falloff_weight(float t) const
  {
    float fac = 0.0f;
    /* Code borrowed from the warp modifier. */
    /* Closely matches PROP_SMOOTH and similar. */
    switch (falloff_type_) {
      case GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF_LINEAR:
        fac = t;
        break;
      case GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF_CURVE:
        fac = BKE_curvemapping_evaluateF(curve_map_, 0, t);
        break;
      case GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF_SHARP:
        fac = t * t;
        break;
      case GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF_SMOOTH:
        fac = 3.0f * t * t - 2.0f * t * t * t;
        break;
      case GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF_ROOT:
        fac = sqrtf(t);
        break;
      case GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF_SPHERE:
        fac = sqrtf(2 * t - t * t);
        break;
      case GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF_STEP:
        fac = (t >= 0.5f) ? 1.0f : 0.0f;
        break;
      default:
        BLI_assert_unreachable();
    }

    return invert_ ? 1.0f - fac : fac;
  }

  template<typename ValueType>
  void add_point(RangeQueryData<ValueType> &data,
                 const float3 &co,
                 float dist_sq,
                 ValueType value) const
  {
    float rel_dist = min_ff(sqrtf(dist_sq) / data.radius_, 1.0f);
    float weight = falloff_weight(1.0f - rel_dist);
    TypeDetails::add_weighted(data.result_, value, weight);
    data.total_weight_ += weight;
  };
};

struct RangeQueryAccumulator_Closest {
  template<typename ValueType>
  void add_point(RangeQueryData<ValueType> &data,
                 const float3 &co,
                 float dist_sq,
                 ValueType value) const
  {
    if (dist_sq < data.min_dist_sq_) {
      data.result_ = value;
      data.total_weight_ = 1.0f;
      data.min_dist_sq_ = dist_sq;
    }
  };
};

struct RangeQueryAccumulator_Min {
  template<typename ValueType>
  void add_point(RangeQueryData<ValueType> &data,
                 const float3 &co,
                 float dist_sq,
                 ValueType value) const
  {
    data.result_ = TypeDetails::min(data.result_, value);
    data.total_weight_ = 1.0f;
  };
};

struct RangeQueryAccumulator_Max {
  template<typename ValueType>
  void add_point(RangeQueryData<ValueType> &data,
                 const float3 &co,
                 float dist_sq,
                 ValueType value) const
  {
    data.result_ = TypeDetails::max(data.result_, value);
    data.total_weight_ = 1.0f;
  };
};

/* Cumulative range query: values, weights and counts are added to current.
 * Caller must ensure these arrays are initialized to zero!
 */
template<typename ValueType, typename AccumulatorType>
static void range_query_bvhtree_typed(const AccumulatorType &accum,
                                      BVHTree *tree,
                                      const VArray<float3> &positions,
                                      const VArray<float> &radii,
                                      const GVArray_Typed<ValueType> &values,
                                      const MutableSpan<ValueType> r_weighted_sums,
                                      const MutableSpan<float> r_total_weights,
                                      const MutableSpan<int> r_counts)
{
  BLI_assert(positions.size() == radii.size() || radii.is_empty());
  BLI_assert(positions.size() == r_weighted_sums.size() || r_weighted_sums.is_empty());
  BLI_assert(positions.size() == r_total_weights.size() || r_total_weights.is_empty());
  BLI_assert(positions.size() == r_counts.size() || r_counts.is_empty());

  IndexRange range = positions.index_range();
  threading::parallel_for(range, 512, [&](IndexRange range) {
    for (const int i : range) {
      const float3 position = positions[i];
      const float radius = radii[i];
      if (radius <= 0.0f) {
        continue;
      }

      RangeQueryUserData<ValueType, AccumulatorType> userdata(accum, values, radius);
      int count = BLI_bvhtree_range_query(tree,
                                          position,
                                          radius,
                                          RangeQueryUserData<ValueType, AccumulatorType>::callback,
                                          &userdata);

      if (!r_weighted_sums.is_empty()) {
        TypeDetails::add(r_weighted_sums[i], userdata.data_.result_);
      }
      if (!r_counts.is_empty()) {
        r_counts[i] += count;
      }
      if (!r_total_weights.is_empty()) {
        r_total_weights[i] += userdata.data_.total_weight_;
      }
    }
  });
}

template<typename AccumulatorType>
static void range_query_bvhtree(const AccumulatorType &accum,
                                BVHTree *tree,
                                const VArray<float3> &positions,
                                const VArray<float> &radii,
                                const GVArrayPtr &values,
                                const GMutableSpan r_weighted_sums,
                                const MutableSpan<float> r_total_weights,
                                const MutableSpan<int> r_counts)
{
  attribute_math::convert_to_static_type(r_weighted_sums.type(), [&](auto dummy) {
    using T = decltype(dummy);
    range_query_bvhtree_typed<T>(accum,
                                 tree,
                                 positions,
                                 radii,
                                 values->typed<T>(),
                                 r_weighted_sums.typed<T>(),
                                 r_total_weights,
                                 r_counts);
  });
}

template<typename AccumulatorType>
static void range_query_add_components(const AccumulatorType &accum,
                                       const GeometrySet &src_geometry,
                                       const StringRef src_name,
                                       CustomDataType data_type,
                                       const VArray<float3> &positions,
                                       const VArray<float> &radii,
                                       const GMutableSpan r_weighted_sums,
                                       const MutableSpan<float> r_total_weights,
                                       const MutableSpan<int> r_counts)
{
  /* If there is a pointcloud, add values from points. */
  const PointCloudComponent *pointcloud_component =
      src_geometry.get_component_for_read<PointCloudComponent>();
  const PointCloud *pointcloud = pointcloud_component ? pointcloud_component->get_for_read() :
                                                        nullptr;
  if (pointcloud != nullptr && pointcloud->totpoint > 0) {
    ReadAttributeLookup src_attribute = pointcloud_component->attribute_try_get_for_read(
        src_name, data_type);
    if (src_attribute) {
      BVHTreeFromPointCloud tree_data;
      BKE_bvhtree_from_pointcloud_get(&tree_data, pointcloud, 2);
      range_query_bvhtree(accum,
                          tree_data.tree,
                          positions,
                          radii,
                          src_attribute.varray,
                          r_weighted_sums,
                          r_total_weights,
                          r_counts);
      free_bvhtree_from_pointcloud(&tree_data);
    }
  }

  /* If there is a mesh, add values from mesh elements. */
  const MeshComponent *mesh_component = src_geometry.get_component_for_read<MeshComponent>();
  const Mesh *mesh = mesh_component ? mesh_component->get_for_read() : nullptr;
  if (mesh != nullptr) {
    ReadAttributeLookup src_attribute = mesh_component->attribute_try_get_for_read(src_name,
                                                                                   data_type);
    if (src_attribute) {
      BVHTreeFromMesh tree_data;
      switch (src_attribute.domain) {
        case ATTR_DOMAIN_POINT: {
          if (mesh->totvert > 0) {
            BKE_bvhtree_from_mesh_get(&tree_data, mesh, BVHTREE_FROM_VERTS, 2);
            range_query_bvhtree(accum,
                                tree_data.tree,
                                positions,
                                radii,
                                src_attribute.varray,
                                r_weighted_sums,
                                r_total_weights,
                                r_counts);
            free_bvhtree_from_mesh(&tree_data);
          }
          break;
        }
        case ATTR_DOMAIN_EDGE: {
          if (mesh->totedge > 0) {
            BKE_bvhtree_from_mesh_get(&tree_data, mesh, BVHTREE_FROM_EDGES, 2);
            range_query_bvhtree(accum,
                                tree_data.tree,
                                positions,
                                radii,
                                src_attribute.varray,
                                r_weighted_sums,
                                r_total_weights,
                                r_counts);
            free_bvhtree_from_mesh(&tree_data);
          }
          break;
        }
        case ATTR_DOMAIN_FACE: {
          if (mesh->totpoly > 0) {
            /* TODO implement triangle merging or only support triangles. This currently crashes without triangulated faces. */
            //BKE_bvhtree_from_mesh_get(&tree_data, mesh, BVHTREE_FROM_FACES, 2);
            //range_query_bvhtree(accum,
            //                    tree_data.tree,
            //                    positions,
            //                    radii,
            //                    src_attribute.varray,
            //                    r_weighted_sums,
            //                    r_total_weights,
            //                    r_counts);
            //free_bvhtree_from_mesh(&tree_data);
          }
          break;
        }
        default: {
          break;
        }
      }
    }
  }
}

static void range_query_normalize(const GMutableSpan weighted_sums,
                                  const Span<float> total_weights)
{
  BLI_assert(total_weights.size() == weighted_sums.size());

  attribute_math::convert_to_static_type(weighted_sums.type(), [&](auto dummy) {
    using T = decltype(dummy);
    const MutableSpan<T> typed_result = weighted_sums.typed<T>();

    threading::parallel_for(IndexRange(typed_result.size()), 512, [&](IndexRange range) {
      for (const int i : range) {
        typed_result[i] = TypeDetails::normalize(typed_result[i], total_weights[i]);
      }
    });
  });
}

static void get_result_domain_and_data_type(const GeometrySet &src_geometry,
                                            const GeometryComponent &dst_component,
                                            const StringRef attribute_name,
                                            CustomDataType *r_data_type,
                                            AttributeDomain *r_domain)
{
  Vector<CustomDataType> data_types;
  Vector<AttributeDomain> domains;

  const PointCloudComponent *pointcloud_component =
      src_geometry.get_component_for_read<PointCloudComponent>();
  if (pointcloud_component != nullptr) {
    std::optional<AttributeMetaData> meta_data = pointcloud_component->attribute_get_meta_data(
        attribute_name);
    if (meta_data.has_value()) {
      data_types.append(meta_data->data_type);
      domains.append(meta_data->domain);
    }
  }

  const MeshComponent *mesh_component = src_geometry.get_component_for_read<MeshComponent>();
  if (mesh_component != nullptr) {
    std::optional<AttributeMetaData> meta_data = mesh_component->attribute_get_meta_data(
        attribute_name);
    if (meta_data.has_value()) {
      data_types.append(meta_data->data_type);
      domains.append(meta_data->domain);
    }
  }

  *r_data_type = bke::attribute_data_type_highest_complexity(data_types);

  if (dst_component.type() == GEO_COMPONENT_TYPE_POINT_CLOUD) {
    *r_domain = ATTR_DOMAIN_POINT;
  }
  else {
    *r_domain = bke::attribute_domain_highest_priority(domains);
  }
}

static void range_query_attribute(const GeoNodeExecParams &params,
                                  const GeometrySet &src_geometry,
                                  GeometryComponent &dst_component,
                                  const StringRef src_name,
                                  const StringRef dst_name,
                                  const StringRef dst_count_name,
                                  const StringRef dst_total_weight_name)
{
  const NodeGeometryAttributeRangeQuery &storage =
      *(const NodeGeometryAttributeRangeQuery *)params.node().storage;
  const GeometryNodeAttributeRangeQueryMode mode = (GeometryNodeAttributeRangeQueryMode)
                                                       storage.mode;
  const AttributeDomain input_domain = (AttributeDomain)storage.domain;

  CustomDataType data_type;
  AttributeDomain auto_domain;
  get_result_domain_and_data_type(src_geometry, dst_component, src_name, &data_type, &auto_domain);
  const AttributeDomain dst_domain = (input_domain == ATTR_DOMAIN_AUTO) ? auto_domain :
                                                                          input_domain;
  const CPPType &cpp_type = *bke::custom_data_type_to_cpp_type(data_type);

  GVArray_Typed<float3> dst_positions = dst_component.attribute_get_for_read<float3>(
      "position", dst_domain, {0, 0, 0});
  GVArray_Typed<float> dst_radii = params.get_input_attribute<float>(
      "Radius", dst_component, dst_domain, 0.0f);
  const int tot_samples = dst_positions.size();

  OutputAttribute dst_attribute = dst_component.attribute_try_get_for_output_only(
      dst_name, dst_domain, data_type);
  if (!dst_attribute) {
    return;
  }
  OutputAttribute dst_counts = dst_component.attribute_try_get_for_output_only(
      dst_count_name, dst_domain, CD_PROP_INT32);
  OutputAttribute dst_total_weights = dst_component.attribute_try_get_for_output_only(
      dst_total_weight_name, dst_domain, CD_PROP_FLOAT);

  Array<int> counts_internal;
  if (!dst_counts) {
    counts_internal.reinitialize(tot_samples);
  }
  Array<float> total_weighs_internal;
  if (!dst_total_weights) {
    total_weighs_internal.reinitialize(tot_samples);
  }
  MutableSpan<int> counts_span = dst_counts ? dst_counts.as_span<int>() : counts_internal;
  MutableSpan<float> total_weights_span = dst_total_weights ? dst_total_weights.as_span<float>() : total_weighs_internal;

  void *output_buffer = MEM_mallocN_aligned(
      tot_samples * cpp_type.size(), cpp_type.alignment(), "weighted_sums");
  GMutableSpan output_span(cpp_type, output_buffer, tot_samples);

  attribute_math::convert_to_static_type(cpp_type, [&](auto dummy) {
    using T = decltype(dummy);
    static const T zero(0);
    output_span.typed<T>().fill(zero);
  });
  total_weights_span.fill(0.0f);
  counts_span.fill(0);

  auto do_range_query = [&](auto accum) {
    range_query_add_components(accum,
                               src_geometry,
                               src_name,
                               data_type,
                               dst_positions,
                               dst_radii,
                               output_span,
                               total_weights_span,
                               counts_span);
  };
  switch ((GeometryNodeAttributeRangeQueryMode)storage.mode) {
    case GEO_NODE_ATTRIBUTE_RANGE_QUERY_AVERAGE:
      do_range_query(RangeQueryAccumulator_Average());
      break;
    case GEO_NODE_ATTRIBUTE_RANGE_QUERY_SUM:
      do_range_query(RangeQueryAccumulator_Sum());
      break;
    case GEO_NODE_ATTRIBUTE_RANGE_QUERY_FALLOFF: {
      GeometryNodeAttributeRangeQueryFalloffType falloff_type =
          (GeometryNodeAttributeRangeQueryFalloffType)storage.falloff_type;
      bool invert = params.node().custom1 & GEO_NODE_ATTRIBUTE_RANGE_QUERY_INVERT_FALLOFF;
      do_range_query(RangeQueryAccumulator_Falloff(falloff_type, invert, storage.falloff_curve));
      break;
    }
    case GEO_NODE_ATTRIBUTE_RANGE_QUERY_CLOSEST:
      do_range_query(RangeQueryAccumulator_Closest());
      break;
    case GEO_NODE_ATTRIBUTE_RANGE_QUERY_MINIMUM:
      do_range_query(RangeQueryAccumulator_Min());
      break;
    case GEO_NODE_ATTRIBUTE_RANGE_QUERY_MAXIMUM:
      do_range_query(RangeQueryAccumulator_Max());
      break;
  }

  /* Normalize by dividing by cumulative weight. */
  range_query_normalize(output_span, total_weights_span);

  for (int i : IndexRange(tot_samples)) {
    dst_attribute->set_by_copy(i, output_span[i]);
  }
  MEM_freeN(output_buffer);

  dst_attribute.save();
  dst_counts.save();
  dst_total_weights.save();
}

static void geo_node_attribute_range_query_exec(GeoNodeExecParams params)
{
  GeometrySet dst_geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet src_geometry_set = params.extract_input<GeometrySet>("Source Geometry");
  const std::string src_attribute_name = params.extract_input<std::string>("Source");
  const std::string dst_attribute_name = params.extract_input<std::string>("Destination");
  const std::string dst_count_name = params.extract_input<std::string>("Count");
  const std::string dst_total_weight_name = params.extract_input<std::string>("Total Weight");

  if (src_attribute_name.empty() || dst_attribute_name.empty()) {
    params.set_output("Geometry", dst_geometry_set);
    return;
  }

  dst_geometry_set = bke::geometry_set_realize_instances(dst_geometry_set);
  src_geometry_set = bke::geometry_set_realize_instances(src_geometry_set);

  if (dst_geometry_set.has<MeshComponent>()) {
    range_query_attribute(params,
                          src_geometry_set,
                          dst_geometry_set.get_component_for_write<MeshComponent>(),
                          src_attribute_name,
                          dst_attribute_name,
                          dst_count_name,
                          dst_total_weight_name);
  }
  if (dst_geometry_set.has<PointCloudComponent>()) {
    range_query_attribute(params,
                          src_geometry_set,
                          dst_geometry_set.get_component_for_write<PointCloudComponent>(),
                          src_attribute_name,
                          dst_attribute_name,
                          dst_count_name,
                          dst_total_weight_name);
  }

  params.set_output("Geometry", dst_geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_attribute_range_query()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_ATTRIBUTE_RANGE_QUERY, "Attribute Range Query", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(
      &ntype, geo_node_attribute_range_query_in, geo_node_attribute_range_query_out);
  node_type_init(&ntype, geo_node_attribute_range_query_init);
  node_type_update(&ntype, geo_node_attribute_range_query_update);
  node_type_storage(&ntype,
                    "NodeGeometryAttributeRangeQuery",
                    geo_node_attribute_range_query_free_storage,
                    geo_node_attribute_range_query_copy_storage);
  ntype.geometry_node_execute = blender::nodes::geo_node_attribute_range_query_exec;
  ntype.draw_buttons = geo_node_attribute_range_query_layout;
  nodeRegisterType(&ntype);
}
