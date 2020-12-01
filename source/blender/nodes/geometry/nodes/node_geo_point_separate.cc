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
    {SOCK_FLOAT, N_("Threshold"), 0.5f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_point_instance_out[] = {
    {SOCK_GEOMETRY, N_("Geometry A")},
    {SOCK_GEOMETRY, N_("Geometry B")},
    {-1, ""},
};

namespace blender::nodes {

static void fill_attribute_from_input(ReadAttributePtr input_attribute,
                                      WriteAttributePtr out_attribute_a,
                                      WriteAttributePtr out_attribute_b,
                                      Span<bool> a_or_b)
{
  fn::GSpan in_span = input_attribute->get_span();
  int i_a = 0;
  int i_b = 0;
  for (int i_in : IndexRange(input_attribute->size())) {
    if (a_or_b[i_in]) {
      out_attribute_a->set(i_a, in_span[i_in]);
      i_a++;
    }
    else {
      out_attribute_b->set(i_b, in_span[i_in]);
      i_b++;
    }
  }
}

static void separate_component_attributes(const PointCloudComponent &component,
                                          GeometrySet *out_set_a,
                                          GeometrySet *out_set_b,
                                          const int a_total,
                                          const int b_total,
                                          Span<bool> a_or_b)
{
  /* Start fresh with new pointclouds. */
  out_set_a->replace_pointcloud(BKE_pointcloud_new_nomain(a_total));
  out_set_b->replace_pointcloud(BKE_pointcloud_new_nomain(b_total));
  PointCloudComponent &out_component_a = out_set_a->get_component_for_write<PointCloudComponent>();
  PointCloudComponent &out_component_b = out_set_b->get_component_for_write<PointCloudComponent>();

  Set<std::string> attribute_names = component.attribute_names();
  for (const std::string &name : attribute_names) {
    ReadAttributePtr attribute = component.attribute_try_get_for_read(name);
    BLI_assert(attribute);

    const CustomDataType data_type = bke::cpp_type_to_custom_data_type(attribute->cpp_type());
    const AttributeDomain domain = attribute->domain();
    if (!component.attribute_is_builtin(name)) {
      const bool create_success_a = out_component_a.attribute_try_create(name, domain, data_type);
      const bool create_success_b = out_component_b.attribute_try_create(name, domain, data_type);
      if (!(create_success_a && create_success_b)) {
        /* Since the input attribute is from the same component type,
         * it should always be possible to create the new attributes. */
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

    fill_attribute_from_input(
        std::move(attribute), std::move(out_attribute_a), std::move(out_attribute_b), a_or_b);
  }
}

/**
 * Find total in each new set and find which of the output sets each point will belong to.
 */
static Array<bool> calculate_split(const GeometryComponent &component,
                                   const std::string mask_name,
                                   const float threshold,
                                   int *r_a_total,
                                   int *r_b_total)
{
  /* For now this will always sample the attributes on the point level. */
  const FloatReadAttribute mask_attribute = component.attribute_get_for_read<float>(
      mask_name, ATTR_DOMAIN_POINT, 1.0f);
  Span<float> masks = mask_attribute.get_span();
  const int in_total = masks.size();

  *r_a_total = 0;
  Array<bool> a_or_b(in_total);
  for (int i : masks.index_range()) {
    const bool in_a = masks[i] > threshold;
    a_or_b[i] = in_a;
    if (in_a) {
      *r_a_total += 1;
    }
  }
  *r_b_total = in_total - *r_a_total;

  return a_or_b;
}

/* Much of the attribute code can be handled generically for every geometry component type. */
template<typename Component>
static void separate_component_type(const Component &component,
                                    GeometrySet *out_set_a,
                                    GeometrySet *out_set_b,
                                    const std::string mask_name,
                                    const float threshold)
{
  int a_total;
  int b_total;
  Array<bool> a_or_b = calculate_split(component, mask_name, threshold, &a_total, &b_total);

  separate_component_attributes(
      component, out_set_a, out_set_b, a_total, b_total, a_or_b.as_span());
}

static void geo_node_point_separate_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  GeometrySet out_set_a(geometry_set);
  GeometrySet out_set_b(geometry_set);

  const std::string mask_name = params.extract_input<std::string>("Mask");
  const float threshold = params.extract_input<float>("Threshold");

  if (geometry_set.has<PointCloudComponent>()) {
    separate_component_type<PointCloudComponent>(
        *geometry_set.get_component_for_read<PointCloudComponent>(),
        &out_set_a,
        &out_set_b,
        mask_name,
        threshold);
  }

  params.set_output("Geometry A", std::move(out_set_a));
  params.set_output("Geometry B", std::move(out_set_b));
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
