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

#include "BLI_array.hh"
#include "BLI_disjoint_set.hh"
#include "BLI_task.hh"
#include "BLI_vector.hh"
#include "BLI_vector_set.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "BKE_mesh.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_scale_elements_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>(N_("Geometry")).supported_type(GEO_COMPONENT_TYPE_MESH);
  b.add_input<decl::Bool>(N_("Selection")).default_value(true).hide_value().supports_field();
  b.add_input<decl::Float>(N_("Scale"), "Scale_Float").default_value(1.0f).supports_field();
  b.add_input<decl::Vector>(N_("Scale"), "Scale_Vector")
      .default_value({1.0f, 1.0f, 1.0f})
      .supports_field();
  b.add_input<decl::Vector>(N_("Pivot")).subtype(PROP_TRANSLATION).implicit_field();
  b.add_input<decl::Vector>(N_("X Axis")).default_value({1.0f, 0.0f, 0.0f}).supports_field();
  b.add_input<decl::Vector>(N_("Up")).default_value({0.0f, 0.0f, 1.0f}).supports_field();
  b.add_output<decl::Geometry>(N_("Geometry"));
};

static void node_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiItemR(layout, ptr, "mode", 0, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "uniform", 0, nullptr, ICON_NONE);
}

static void node_init(bNodeTree *UNUSED(tree), bNode *node)
{
  node->custom1 = GEO_NODE_SCALE_ELEMENTS_MODE_FACE;
  node->custom2 = GEO_NODE_SCALE_ELEMENTS_UNIFORM;
}

static void node_update(bNodeTree *ntree, bNode *node)
{
  bNodeSocket *geometry_socket = static_cast<bNodeSocket *>(node->inputs.first);
  bNodeSocket *selection_socket = geometry_socket->next;
  bNodeSocket *scale_float_socket = selection_socket->next;
  bNodeSocket *scale_vector_socket = scale_float_socket->next;
  bNodeSocket *pivot_socket = scale_vector_socket->next;
  bNodeSocket *x_axis_socket = pivot_socket->next;
  bNodeSocket *up_socket = x_axis_socket->next;

  const bool use_uniform_scale = node->custom2 & GEO_NODE_SCALE_ELEMENTS_UNIFORM;

  nodeSetSocketAvailability(ntree, scale_float_socket, use_uniform_scale);
  nodeSetSocketAvailability(ntree, scale_vector_socket, !use_uniform_scale);
  nodeSetSocketAvailability(ntree, x_axis_socket, !use_uniform_scale);
  nodeSetSocketAvailability(ntree, up_socket, !use_uniform_scale);
}

struct InputFields {
  bool use_uniform_scale;
  Field<bool> selection;
  Field<float> uniform_scale;
  Field<float3> vector_scale;
  Field<float3> pivot;
  Field<float3> x_axis;
  Field<float3> up;
};

struct EvaluatedFields {
  IndexMask selection;
  VArray<float> uniform_scales;
  VArray<float3> vector_scales;
  VArray<float3> pivots;
  VArray<float3> x_axis_vectors;
  VArray<float3> up_vectors;
};

struct ScaleIsland {
  /* Either face or edge indices. */
  Vector<int> element_indices;
};

static float4x4 create_transform(const float3 &pivot,
                                 float3 x_axis,
                                 const float3 &up,
                                 const float3 &scale)
{
  x_axis = x_axis.normalized();
  const float3 y_axis = -float3::cross(x_axis, up).normalized();
  const float3 z_axis = float3::cross(x_axis, y_axis);

  float4x4 transform;
  unit_m4(transform.values);
  sub_v3_v3(transform.values[3], pivot);

  float4x4 axis_transform;
  unit_m4(axis_transform.values);
  copy_v3_v3(axis_transform.values[0], x_axis);
  copy_v3_v3(axis_transform.values[1], y_axis);
  copy_v3_v3(axis_transform.values[2], z_axis);

  float4x4 axis_transform_inv = axis_transform.transposed();

  float4x4 scale_transform;
  unit_m4(scale_transform.values);
  scale_transform.values[0][0] = scale.x;
  scale_transform.values[1][1] = scale.y;
  scale_transform.values[2][2] = scale.z;

  transform = axis_transform * scale_transform * axis_transform_inv * transform;
  add_v3_v3(transform.values[3], pivot);

  return transform;
}

static EvaluatedFields evaluate_fields(FieldEvaluator &evaluator, const InputFields &input_fields)
{
  EvaluatedFields evaluated;
  evaluator.set_selection(input_fields.selection);
  if (input_fields.use_uniform_scale) {
    evaluator.add(input_fields.uniform_scale, &evaluated.uniform_scales);
  }
  else {
    evaluator.add(input_fields.vector_scale, &evaluated.vector_scales);
    evaluator.add(input_fields.x_axis, &evaluated.x_axis_vectors);
    evaluator.add(input_fields.up, &evaluated.up_vectors);
  }
  evaluator.add(input_fields.pivot, &evaluated.pivots);
  evaluator.evaluate();
  evaluated.selection = evaluator.get_evaluated_selection_as_mask();
  return evaluated;
}

static void scale_vertex_islands(
    Mesh &mesh,
    const Span<ScaleIsland> islands,
    const EvaluatedFields &evaluated,
    const FunctionRef<void(int element_index, Vector<int> &r_vertex_indices)> get_vertex_indices)
{
  threading::parallel_for(islands.index_range(), 256, [&](const IndexRange range) {
    Set<int> handled_vertices;
    for (const int island_index : range) {
      const ScaleIsland &island = islands[island_index];

      float3 scale = {0.0f, 0.0f, 0.0f};
      float3 pivot = {0.0f, 0.0f, 0.0f};
      float3 x_axis = {0.0f, 0.0f, 0.0f};
      float3 up = {0.0f, 0.0f, 0.0f};

      Vector<int> vertex_indices;
      for (const int poly_index : island.element_indices) {
        get_vertex_indices(poly_index, vertex_indices);
        pivot += evaluated.pivots[poly_index];
        if (evaluated.uniform_scales) {
          scale += float3(evaluated.uniform_scales[poly_index]);
          x_axis += float3(1, 0, 0);
          up += float3(0, 0, 1);
        }
        else {
          scale += evaluated.vector_scales[poly_index];
          x_axis += evaluated.x_axis_vectors[poly_index];
          up += evaluated.up_vectors[poly_index];
        }
      }

      const float f = 1.0f / island.element_indices.size();
      scale *= f;
      pivot *= f;
      x_axis *= f;
      up *= f;

      const float4x4 transform = create_transform(pivot, x_axis, up, scale);
      handled_vertices.clear();
      for (const int vert_index : vertex_indices) {
        if (!handled_vertices.add(vert_index)) {
          continue;
        }
        MVert &vert = mesh.mvert[vert_index];
        const float3 old_position = vert.co;
        const float3 new_position = transform * old_position;
        copy_v3_v3(vert.co, new_position);
      }
    }
  });

  BKE_mesh_normals_tag_dirty(&mesh);
}

static Vector<ScaleIsland> prepare_face_islands(const Mesh &mesh, const IndexMask face_selection)
{
  DisjointSet disjoint_set(mesh.totvert);
  for (const int poly_index : face_selection) {
    const MPoly &poly = mesh.mpoly[poly_index];
    const Span<MLoop> poly_loops{mesh.mloop + poly.loopstart, poly.totloop};
    for (const int loop_index : IndexRange(poly.totloop - 1)) {
      const int v1 = poly_loops[loop_index].v;
      const int v2 = poly_loops[loop_index + 1].v;
      disjoint_set.join(v1, v2);
    }
    disjoint_set.join(poly_loops.first().v, poly_loops.last().v);
  }

  VectorSet<int> island_ids;
  Vector<ScaleIsland> islands;
  islands.reserve(face_selection.size());
  for (const int poly_index : face_selection) {
    const MPoly &poly = mesh.mpoly[poly_index];
    const Span<MLoop> poly_loops{mesh.mloop + poly.loopstart, poly.totloop};
    const int island_id = disjoint_set.find_root(poly_loops[0].v);
    const int island_index = island_ids.index_of_or_add(island_id);
    if (island_index == islands.size()) {
      islands.append_as();
    }
    ScaleIsland &island = islands[island_index];
    island.element_indices.append(poly_index);
  }

  return islands;
}

static void scale_faces(MeshComponent &mesh_component, const InputFields &input_fields)
{
  Mesh &mesh = *mesh_component.get_for_write();
  mesh.mvert = static_cast<MVert *>(
      CustomData_duplicate_referenced_layer(&mesh.vdata, CD_MVERT, mesh.totvert));

  GeometryComponentFieldContext field_context{mesh_component, ATTR_DOMAIN_FACE};
  FieldEvaluator evaluator{field_context, mesh.totpoly};
  EvaluatedFields evaluated = evaluate_fields(evaluator, input_fields);

  Vector<ScaleIsland> island = prepare_face_islands(mesh, evaluated.selection);
  scale_vertex_islands(
      mesh, island, evaluated, [&](int face_index, Vector<int> &r_vertex_indices) {
        const MPoly &poly = mesh.mpoly[face_index];
        const Span<MLoop> poly_loops{mesh.mloop + poly.loopstart, poly.totloop};
        for (const MLoop &loop : poly_loops) {
          r_vertex_indices.append(loop.v);
        }
      });
}

static Vector<ScaleIsland> prepare_edge_islands(const Mesh &mesh, const IndexMask edge_selection)
{
  DisjointSet disjoint_set(mesh.totvert);
  for (const int edge_index : edge_selection) {
    const MEdge &edge = mesh.medge[edge_index];
    disjoint_set.join(edge.v1, edge.v2);
  }

  VectorSet<int> island_ids;
  Vector<ScaleIsland> islands;
  islands.reserve(edge_selection.size());
  for (const int edge_index : edge_selection) {
    const MEdge &edge = mesh.medge[edge_index];
    const int island_id = disjoint_set.find_root(edge.v1);
    const int island_index = island_ids.index_of_or_add(island_id);
    if (island_index == islands.size()) {
      islands.append_as();
    }
    ScaleIsland &island = islands[island_index];
    island.element_indices.append(edge_index);
  }

  return islands;
}

static void scale_edges(MeshComponent &mesh_component, const InputFields &input_fields)
{
  Mesh &mesh = *mesh_component.get_for_write();
  mesh.mvert = static_cast<MVert *>(
      CustomData_duplicate_referenced_layer(&mesh.vdata, CD_MVERT, mesh.totvert));

  GeometryComponentFieldContext field_context{mesh_component, ATTR_DOMAIN_EDGE};
  FieldEvaluator evaluator{field_context, mesh.totedge};
  EvaluatedFields evaluated = evaluate_fields(evaluator, input_fields);

  Vector<ScaleIsland> island = prepare_edge_islands(mesh, evaluated.selection);
  scale_vertex_islands(
      mesh, island, evaluated, [&](const int edge_index, Vector<int> &r_vertex_indices) {
        const MEdge &edge = mesh.medge[edge_index];
        r_vertex_indices.append(edge.v1);
        r_vertex_indices.append(edge.v2);
      });
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const bNode &node = params.node();
  const GeometryNodeScaleElementsMode mode = static_cast<GeometryNodeScaleElementsMode>(
      node.custom1);

  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry");
  InputFields input_fields;
  input_fields.use_uniform_scale = node.custom2 & GEO_NODE_SCALE_ELEMENTS_UNIFORM;
  input_fields.selection = params.get_input<Field<bool>>("Selection");
  if (input_fields.use_uniform_scale) {
    input_fields.uniform_scale = params.get_input<Field<float>>("Scale_Float");
  }
  else {
    input_fields.vector_scale = params.get_input<Field<float3>>("Scale_Vector");
    input_fields.x_axis = params.get_input<Field<float3>>("X Axis");
    input_fields.up = params.get_input<Field<float3>>("Up");
  }
  input_fields.pivot = params.get_input<Field<float3>>("Pivot");

  geometry.modify_geometry_sets([&](GeometrySet &geometry) {
    switch (mode) {
      case GEO_NODE_SCALE_ELEMENTS_MODE_FACE: {
        if (geometry.has_mesh()) {
          MeshComponent &mesh_component = geometry.get_component_for_write<MeshComponent>();
          scale_faces(mesh_component, input_fields);
        }
        break;
      }
      case GEO_NODE_SCALE_ELEMENTS_MODE_EDGE: {
        if (geometry.has_mesh()) {
          MeshComponent &mesh_component = geometry.get_component_for_write<MeshComponent>();
          scale_edges(mesh_component, input_fields);
        }
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
  ntype.draw_buttons = file_ns::node_layout;
  ntype.initfunc = file_ns::node_init;
  ntype.updatefunc = file_ns::node_update;
  nodeRegisterType(&ntype);
}
