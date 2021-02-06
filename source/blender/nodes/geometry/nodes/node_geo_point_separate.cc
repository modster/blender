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

#include "BKE_mesh.h"
#include "BKE_persistent_data_handle.hh"
#include "BKE_pointcloud.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_pointcloud_types.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_point_instance_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_STRING, N_("Mask")},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_point_instance_out[] = {
    {SOCK_GEOMETRY, N_("Geometry 1")},
    {SOCK_GEOMETRY, N_("Geometry 2")},
    {-1, ""},
};

namespace blender::nodes {

static void fill_new_attribute_from_input(const ReadAttribute &input_attribute,
                                          WriteAttribute &out_attribute_a,
                                          WriteAttribute &out_attribute_b,
                                          Span<bool> a_or_b)
{
  fn::GSpan in_span = input_attribute.get_span();
  int i_a = 0;
  int i_b = 0;
  for (int i_in = 0; i_in < in_span.size(); i_in++) {
    const bool move_to_b = a_or_b[i_in];
    if (move_to_b) {
      out_attribute_b.set(i_b, in_span[i_in]);
      i_b++;
    }
    else {
      out_attribute_a.set(i_a, in_span[i_in]);
      i_a++;
    }
  }
}

/**
 * Move the original attribute values to the two output components.
 *
 * \note This assumes a consistent ordering of indices before and after the split,
 * which is true for points and a simple vertex array.
 */
static void move_split_attributes(const GeometryComponent &in_component,
                                  GeometryComponent &out_component_a,
                                  GeometryComponent &out_component_b,
                                  Span<bool> a_or_b)
{
  Set<std::string> attribute_names = in_component.attribute_names();

  for (const std::string &name : attribute_names) {
    ReadAttributePtr attribute = in_component.attribute_try_get_for_read(name);
    BLI_assert(attribute);

    /* Since this node only creates points and vertices, don't copy other attributes. */
    if (attribute->domain() != ATTR_DOMAIN_POINT) {
      continue;
    }

    const CustomDataType data_type = bke::cpp_type_to_custom_data_type(attribute->cpp_type());
    const AttributeDomain domain = attribute->domain();

    /* Don't try to create the attribute on the new component if it already exists. Built-in
     * attributes will already exist on new components by definition. It should always be possible
     * to recreate the attribute on the same component type. Also, if one of the new components
     * has the attribute the other one should have it too, but check independently to be safe. */
    if (!out_component_a.attribute_exists(name)) {
      if (!out_component_a.attribute_try_create(name, domain, data_type)) {
        BLI_assert(false);
        continue;
      }
    }
    if (!out_component_b.attribute_exists(name)) {
      if (!out_component_b.attribute_try_create(name, domain, data_type)) {
        BLI_assert(false);
        continue;
      }
    }

    WriteAttributePtr out_attribute_a = out_component_a.attribute_try_get_for_write(name);
    WriteAttributePtr out_attribute_b = out_component_b.attribute_try_get_for_write(name);
    if (!out_attribute_a || !out_attribute_b) {
      BLI_assert(false);
      continue;
    }

    fill_new_attribute_from_input(*attribute, *out_attribute_a, *out_attribute_b, a_or_b);
  }
}

static void gather_component_attribute_info(const GeometryComponent &component,
                                            Map<std::string, CustomDataType> &attribute_info,
                                            Set<std::string> ignored_attributes)
{
  for (const std::string name : component.attribute_names()) {
    if (ignored_attributes.contains(name)) {
      continue;
    }
    const ReadAttributePtr read_attribute = component.attribute_try_get_for_read(
        name, ATTR_DOMAIN_POINT);
    if (!read_attribute) {
      continue;
    }
    const CustomDataType data_type = read_attribute->custom_data_type();
    attribute_info.add_or_modify(
        name,
        [&data_type](CustomDataType *final_data_type) { *final_data_type = data_type; },
        [&data_type](CustomDataType *final_data_type) {
          *final_data_type = attribute_data_type_highest_complexity({*final_data_type, data_type});
        });
  }
}

static void gather_component_positions(Span<float3> positions,
                                       Span<bool> masks,
                                       const float4x4 &transform,
                                       Vector<float3> &r_positions_a,
                                       Vector<float3> &r_positions_b)
{
  for (const int i : positions.index_range()) {
    if (masks[i]) {
      r_positions_b.append(transform * positions[i]);
    }
    else {
      r_positions_a.append(transform * positions[i]);
    }
  }
}

static void copy_attributes(const int offset,
                            Span<bool> masks,
                            Map<std::string, CustomDataType> &attribute_info,
                            GeometryComponent &out_component_a,
                            GeometryComponent &out_component_b)
{
  for (Map<std::string, CustomDataType>::Item entry : attribute_info.items()) {
    StringRef name = entry.key;
    const CustomDataType data_type_output = entry.value;
  }
}

/**
 * \note This could be relatively easily optimized for the case where an entire
 * component does not have the mask attribute, and thus is moved only to the "a" output.
 */
static void geo_node_point_separate_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet out_set_a(geometry_set);
  GeometrySet out_set_b;

  Map<std::string, CustomDataType> attribute_info;

  Vector<GeometryInstanceGroup> set_groups = BKE_geometry_set_gather_instanced(geometry_set);

  const std::string mask_attribute_name = params.extract_input<std::string>("Mask");
  Vector<float3> positions_a;
  Vector<float3> positions_b;
  BKE_geometry_set_foreach_component_recursive(
      geometry_set, [&](const GeometryComponent &component, Span<float4x4> transforms) {
        const int domain_size = component.attribute_domain_size(ATTR_DOMAIN_POINT);
        if (domain_size == 0) {
          return;
        }

        gather_component_attribute_info(component, attribute_info, {"position"});

        /* Move positions to either output. */
        const Float3ReadAttribute positions_attribute = component.attribute_get_for_read<float3>(
            "position", ATTR_DOMAIN_POINT, {0.0f, 0.0f, 0.0f});
        const BooleanReadAttribute mask_attribute = component.attribute_get_for_read<bool>(
            mask_attribute_name, ATTR_DOMAIN_POINT, false);

        for (const float4x4 &transform : transforms) {
          gather_component_positions(positions_attribute.get_span(),
                                     mask_attribute.get_span(),
                                     transform,
                                     positions_a,
                                     positions_b);
        }
      });

  PointCloudComponent &out_component_a = out_set_a.get_component_for_write<PointCloudComponent>();
  PointCloudComponent &out_component_b = out_set_b.get_component_for_write<PointCloudComponent>();
  PointCloud *pointcloud_a = BKE_pointcloud_new_nomain(positions_a.size());
  PointCloud *pointcloud_b = BKE_pointcloud_new_nomain(positions_b.size());
  memcpy(pointcloud_a->co, positions_a.data(), positions_a.end() - positions_a.begin());
  memcpy(pointcloud_b->co, positions_b.data(), positions_b.end() - positions_b.begin());
  out_component_a.replace(pointcloud_a);
  out_component_b.replace(pointcloud_b);

  /* Create attributes on output components. */
  for (Map<std::string, CustomDataType>::Item entry : attribute_info.items()) {
    StringRef name = entry.key;
    const CustomDataType data_type = entry.value;
    out_component_a.attribute_try_create(name, ATTR_DOMAIN_POINT, data_type);
    out_component_b.attribute_try_create(name, ATTR_DOMAIN_POINT, data_type);
  }

  /* Note: The following code is not ideal for a few reasons:
   * 1. It will repeat the same mask checks for every instance of a component,
   * when the result is going to be the same every time.
   * 2. It retrieves the write attributes from the output components once for every
   * input geometry component. Every retrieval will have overhead. */

  int offset = 0;
  BKE_geometry_set_foreach_component_recursive(
      geometry_set, [&](const GeometryComponent &component, Span<float4x4> transforms) {
        const int domain_size = component.attribute_domain_size(ATTR_DOMAIN_POINT);
        if (domain_size == 0) {
          return;
        }

        const BooleanReadAttribute mask_attribute = component.attribute_get_for_read<bool>(
            mask_attribute_name, ATTR_DOMAIN_POINT, false);
        Span<bool> masks = mask_attribute.get_span();

        // for (Map<std::string, CustomDataType>::Item entry : attribute_info.items()) {
        //   copy_attribute_to_outputs()
        //   StringRef name = entry.key;
        //   const CustomDataType data_type = entry.value;
        //   ReadAttributePtr output_attribute_a = out_component_a.attribute_try_get_for_read(
        //       name, ATTR_DOMAIN_POINT, data_type);
        //   ReadAttributePtr output_attribute_b = out_component_b.attribute_try_get_for_read(
        //       name, ATTR_DOMAIN_POINT, data_type);

        //   for (const int UNUSED(i) : transforms.index_range()) {
        //     copy_attribute(offset, masks, attribute_info, out_component_a, out_component_b);
        //     offset += domain_size;
        //   }
        // }
      });

  params.set_output("Geometry 1", std::move(out_set_a));
  params.set_output("Geometry 2", std::move(out_set_b));
}

}  // namespace blender::nodes

void register_node_type_geo_point_separate()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_POINT_SEPARATE, "Point Separate", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_point_instance_in, geo_node_point_instance_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_point_separate_exec;
  nodeRegisterType(&ntype);
}
