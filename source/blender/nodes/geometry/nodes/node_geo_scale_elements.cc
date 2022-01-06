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

#include "BLI_disjoint_set.hh"
#include "BLI_task.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_scale_elements_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>(N_("Geometry")).supported_type(GEO_COMPONENT_TYPE_MESH);
  b.add_input<decl::Bool>(N_("Selection")).default_value(true).hide_value().supports_field();
  b.add_input<decl::Vector>(N_("Scale")).default_value({1.0f, 1.0f, 1.0f}).supports_field();
  b.add_input<decl::Vector>(N_("Pivot")).subtype(PROP_TRANSLATION).implicit_field();
  b.add_output<decl::Geometry>(N_("Geometry"));
};

static void scale_faces(MeshComponent &mesh_component,
                        const Field<bool> &selection_field,
                        const Field<float3> &scale_field,
                        const Field<float3> &pivot_field)
{
  Mesh *mesh = mesh_component.get_for_write();
  mesh->mvert = static_cast<MVert *>(
      CustomData_duplicate_referenced_layer(&mesh->vdata, CD_MVERT, mesh->totvert));

  GeometryComponentFieldContext field_context{mesh_component, ATTR_DOMAIN_FACE};
  FieldEvaluator evaluator{field_context, mesh->totpoly};
  evaluator.set_selection(selection_field);
  evaluator.add(scale_field);
  evaluator.add(pivot_field);
  evaluator.evaluate();
  const IndexMask selection = evaluator.get_evaluated_selection_as_mask();
  const VArray<float3> scales = evaluator.get_evaluated<float3>(0);
  const VArray<float3> pivots = evaluator.get_evaluated<float3>(1);

  DisjointSet disjoint_set(mesh->totvert);
  for (const int poly_index : selection) {
    const MPoly &poly = mesh->mpoly[poly_index];
    const Span<MLoop> poly_loops{mesh->mloop + poly.loopstart, poly.totloop};
    for (const int loop_index : IndexRange(poly.totloop - 1)) {
      const int v1 = poly_loops[loop_index].v;
      const int v2 = poly_loops[loop_index + 1].v;
      disjoint_set.join(v1, v2);
    }
    disjoint_set.join(poly_loops.first().v, poly_loops.last().v);
  }

  const Span<int64_t> group_by_vertex_index = disjoint_set.ensure_all_roots();

  struct GroupData {
    float3 scale = {0.0f, 0.0f, 0.0f};
    float3 pivot = {0.0f, 0.0f, 0.0f};
    int tot_faces = 0;
  };

  const int max_group_index = mesh->totvert;
  Array<GroupData> groups_data(max_group_index);
  for (const int poly_index : selection) {
    const MPoly &poly = mesh->mpoly[poly_index];
    const int first_vertex = mesh->mloop[poly.loopstart].v;
    const int group_index = group_by_vertex_index[first_vertex];
    const float3 scale = scales[poly_index];
    const float3 pivot = pivots[poly_index];
    GroupData &group_info = groups_data[group_index];
    group_info.pivot += pivot;
    group_info.scale += scale;
    group_info.tot_faces++;
  }

  for (GroupData &group_data : groups_data) {
    if (group_data.tot_faces >= 2) {
      const float f = 1.0f / group_data.tot_faces;
      group_data.scale *= f;
      group_data.pivot *= f;
    }
  }

  threading::parallel_for(IndexRange(mesh->totvert), 1024, [&](const IndexRange range) {
    for (const int vert_index : range) {
      const int group_index = group_by_vertex_index[vert_index];
      const GroupData &group_data = groups_data[group_index];
      if (group_data.tot_faces == 0) {
        continue;
      }
      MVert &vert = mesh->mvert[vert_index];
      const float3 diff = float3(vert.co) - group_data.pivot;
      const float3 new_diff = diff * group_data.scale;
      copy_v3_v3(vert.co, group_data.pivot + new_diff);
    }
  });
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const GeometryNodeScaleElementsMode mode = static_cast<GeometryNodeScaleElementsMode>(
      params.node().custom1);

  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry");
  const Field<bool> &selection_field = params.get_input<Field<bool>>("Selection");
  const Field<float3> &scale_field = params.get_input<Field<float3>>("Scale");
  const Field<float3> &pivot_field = params.get_input<Field<float3>>("Pivot");

  geometry.modify_geometry_sets([&](GeometrySet &geometry) {
    switch (mode) {
      case GEO_NODE_SCALE_ELEMENTS_MODE_FACE: {
        if (geometry.has_mesh()) {
          MeshComponent &mesh_component = geometry.get_component_for_write<MeshComponent>();
          scale_faces(mesh_component, selection_field, scale_field, pivot_field);
        }
        break;
      }
      case GEO_NODE_SCALE_ELEMENTS_MODE_EDGE: {
        break;
      }
      case GEO_NODE_SCALE_ELEMENTS_MODE_CURVE: {
        break;
      }
    }
  });

  params.set_output("Geometry", std::move(geometry));
}

}  // namespace blender::nodes::node_geo_scale_elements_cc

void register_node_type_geo_scale_elements()
{
  namespace file_ns = blender::nodes::node_geo_scale_elements_cc;

  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_SCALE_ELEMENTS, "Scale Elements", NODE_CLASS_GEOMETRY);
  ntype.geometry_node_execute = file_ns::node_geo_exec;
  ntype.declare = file_ns::node_declare;
  nodeRegisterType(&ntype);
}
