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

#include "BLI_task.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_attribute_math.hh"
#include "BKE_mesh.h"
#include "BKE_mesh_runtime.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_extrude_mesh_cc {

NODE_STORAGE_FUNCS(NodeGeometryExtrudeMesh)

/* TODO: Decide whether to transfer attributes by topology proximity to new faces, corners, and
 * edges. */
/* TODO: Deduplicate edge extrusion between edge and face modes. */

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Mesh").supported_type(GEO_COMPONENT_TYPE_MESH);
  b.add_input<decl::Bool>(N_("Selection")).default_value(true).supports_field().hide_value();
  b.add_input<decl::Vector>(N_("Offset")).supports_field().subtype(PROP_TRANSLATION);
  b.add_output<decl::Geometry>("Mesh");
  b.add_output<decl::Bool>(N_("Top")).field_source();
  b.add_output<decl::Bool>(N_("Side")).field_source();
}

static void node_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "mode", 0, "", ICON_NONE);
}

static void node_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometryExtrudeMesh *data = MEM_cnew<NodeGeometryExtrudeMesh>(__func__);
  data->mode = GEO_NODE_EXTRUDE_MESH_FACES;
  node->storage = data;
}

struct AttributeOutputs {
  StrongAnonymousAttributeID top_id;
  StrongAnonymousAttributeID side_id;
};

static void save_selection_as_attribute(MeshComponent &component,
                                        const AnonymousAttributeID *id,
                                        const AttributeDomain domain,
                                        const IndexMask selection)
{
  BLI_assert(!component.attribute_exists(id));

  OutputAttribute_Typed<bool> attribute = component.attribute_try_get_for_output_only<bool>(
      id, domain);
  if (selection.is_range()) {
    attribute.as_span().slice(selection.as_range()).fill(true);
  }
  else {
    attribute.as_span().fill_indices(selection, true);
  }

  attribute.save();
}

static void expand_mesh_size(Mesh &mesh,
                             const int vert_expand,
                             const int edge_expand,
                             const int poly_expand,
                             const int loop_expand)
{
  if (vert_expand != 0) {
    mesh.totvert += vert_expand;
    CustomData_duplicate_referenced_layers(&mesh.vdata, mesh.totvert);
    CustomData_realloc(&mesh.vdata, mesh.totvert);
  }
  else {
    /* Even when the number of vertices is not changed, the mesh can still be deformed. */
    CustomData_duplicate_referenced_layer(&mesh.vdata, CD_MVERT, mesh.totvert);
  }
  if (edge_expand != 0) {
    mesh.totedge += edge_expand;
    CustomData_duplicate_referenced_layers(&mesh.edata, mesh.totedge);
    CustomData_realloc(&mesh.edata, mesh.totedge);
  }
  if (poly_expand != 0) {
    mesh.totpoly += poly_expand;
    CustomData_duplicate_referenced_layers(&mesh.pdata, mesh.totpoly);
    CustomData_realloc(&mesh.pdata, mesh.totpoly);
  }
  if (loop_expand != 0) {
    mesh.totloop += loop_expand;
    CustomData_duplicate_referenced_layers(&mesh.ldata, mesh.totloop);
    CustomData_realloc(&mesh.ldata, mesh.totloop);
  }
  BKE_mesh_update_customdata_pointers(&mesh, false);
}

static void extrude_mesh_vertices(MeshComponent &component,
                                  const Field<bool> &selection_field,
                                  const Field<float3> &offset_field,
                                  const AttributeOutputs &attribute_outputs)
{
  Mesh &mesh = *component.get_for_write();
  const int orig_vert_size = mesh.totvert;
  const int orig_edge_size = mesh.totedge;

  GeometryComponentFieldContext context{component, ATTR_DOMAIN_POINT};
  FieldEvaluator evaluator{context, mesh.totvert};
  evaluator.add(offset_field);
  evaluator.set_selection(selection_field);
  evaluator.evaluate();
  const IndexMask selection = evaluator.get_evaluated_selection_as_mask();
  const VArray<float3> offsets = evaluator.get_evaluated<float3>(0);

  expand_mesh_size(mesh, selection.size(), selection.size(), 0, 0);

  const IndexRange new_vert_range{orig_vert_size, selection.size()};
  const IndexRange new_edge_range{orig_edge_size, selection.size()};

  MutableSpan<MVert> new_verts = bke::mesh_verts(mesh).slice(new_vert_range);
  MutableSpan<MEdge> new_edges = bke::mesh_edges(mesh).slice(new_edge_range);

  for (const int i_selection : selection.index_range()) {
    MEdge &edge = new_edges[i_selection];
    edge.v1 = selection[i_selection];
    edge.v2 = orig_vert_size + i_selection;
    edge.flag = ME_LOOSEEDGE;
  }

  component.attribute_foreach([&](const AttributeIDRef &id, const AttributeMetaData meta_data) {
    if (meta_data.domain == ATTR_DOMAIN_POINT) {
      OutputAttribute attribute = component.attribute_try_get_for_output(
          id, ATTR_DOMAIN_POINT, meta_data.data_type);

      attribute_math::convert_to_static_type(meta_data.data_type, [&](auto dummy) {
        using T = decltype(dummy);
        MutableSpan<T> data = attribute.as_span().typed<T>();
        MutableSpan<T> new_data = data.take_back(selection.size());

        for (const int i : selection.index_range()) {
          new_data[i] = data[selection[i]];
        }
      });

      attribute.save();
    }
    return true;
  });

  devirtualize_varray(offsets, [&](const auto offsets) {
    threading::parallel_for(selection.index_range(), 1024, [&](const IndexRange range) {
      for (const int i : range) {
        const float3 offset = offsets[selection[i]];
        add_v3_v3(new_verts[i].co, offset);
      }
    });
  });

  if (attribute_outputs.top_id) {
    save_selection_as_attribute(
        component, attribute_outputs.top_id.get(), ATTR_DOMAIN_POINT, new_vert_range);
  }
  if (attribute_outputs.side_id) {
    save_selection_as_attribute(
        component, attribute_outputs.side_id.get(), ATTR_DOMAIN_EDGE, new_edge_range);
  }

  BKE_mesh_runtime_clear_cache(&mesh);
  BKE_mesh_normals_tag_dirty(&mesh);

  BKE_mesh_calc_normals(component.get_for_write());
  BLI_assert(BKE_mesh_is_valid(component.get_for_write()));
}

static void extrude_mesh_edges(MeshComponent &component,
                               const Field<bool> &selection_field,
                               const Field<float3> &offset_field,
                               const AttributeOutputs &attribute_outputs)
{
  Mesh &mesh = *component.get_for_write();
  const int orig_vert_size = mesh.totvert;
  Span<MEdge> orig_edges{mesh.medge, mesh.totedge};
  Span<MPoly> orig_polys{mesh.mpoly, mesh.totpoly};
  const int orig_loop_size = mesh.totloop;

  GeometryComponentFieldContext edge_context{component, ATTR_DOMAIN_EDGE};
  FieldEvaluator edge_evaluator{edge_context, mesh.totedge};
  edge_evaluator.add(selection_field);
  edge_evaluator.evaluate();
  const IndexMask selection = edge_evaluator.get_evaluated_as_mask(0);

  /* Maps vertex indices in the original mesh to the corresponding extruded vertices. */
  Array<int> extrude_vert_indices(mesh.totvert, -1);
  /* Maps from the index in the added vertices to the original vertex they were extruded from. */
  Vector<int> extrude_vert_orig_indices;
  extrude_vert_orig_indices.reserve(selection.size());
  for (const int i_edge : selection) {
    const MEdge &edge = orig_edges[i_edge];

    if (extrude_vert_indices[edge.v1] == -1) {
      extrude_vert_indices[edge.v1] = orig_vert_size + extrude_vert_orig_indices.size();
      extrude_vert_orig_indices.append(edge.v1);
    }

    if (extrude_vert_indices[edge.v2] == -1) {
      extrude_vert_indices[edge.v2] = orig_vert_size + extrude_vert_orig_indices.size();
      extrude_vert_orig_indices.append(edge.v2);
    }
  }

  Array<float3> offsets(orig_vert_size);
  GeometryComponentFieldContext point_context{component, ATTR_DOMAIN_POINT};
  FieldEvaluator point_evaluator{point_context, orig_vert_size}; /* TODO: Better selection. */
  point_evaluator.add_with_destination(offset_field, offsets.as_mutable_span());
  point_evaluator.evaluate();

  const IndexRange extrude_vert_range{orig_vert_size, extrude_vert_orig_indices.size()};
  const IndexRange extrude_edge_range{orig_edges.size(), extrude_vert_range.size()};
  const IndexRange duplicate_edge_range{extrude_edge_range.one_after_last(), selection.size()};
  const int new_poly_size = selection.size();
  const int new_loop_size = new_poly_size * 4;

  expand_mesh_size(mesh,
                   extrude_vert_range.size(),
                   extrude_edge_range.size() + duplicate_edge_range.size(),
                   new_poly_size,
                   new_loop_size);

  MutableSpan<MVert> verts{mesh.mvert, mesh.totvert};
  MutableSpan<MVert> new_verts = verts.slice(extrude_vert_range);
  MutableSpan<MEdge> edges{mesh.medge, mesh.totedge};
  MutableSpan<MEdge> extrude_edges = edges.slice(extrude_edge_range);
  MutableSpan<MEdge> duplicate_edges = edges.slice(duplicate_edge_range);
  MutableSpan<MPoly> polys{mesh.mpoly, mesh.totpoly};
  MutableSpan<MPoly> new_polys = polys.take_back(selection.size());
  MutableSpan<MLoop> loops{mesh.mloop, mesh.totloop};
  MutableSpan<MLoop> new_loops = loops.take_back(new_loop_size);

  for (MVert &vert : new_verts) {
    vert.flag = 0;
  }

  for (const int i : extrude_edges.index_range()) {
    MEdge &edge = extrude_edges[i];
    edge.v1 = extrude_vert_orig_indices[i];
    edge.v2 = orig_vert_size + i;
    edge.flag = (ME_EDGEDRAW | ME_EDGERENDER);
  }

  for (const int i : duplicate_edges.index_range()) {
    const MEdge &orig_edge = mesh.medge[selection[i]];
    MEdge &edge = duplicate_edges[i];
    edge.v1 = extrude_vert_indices[orig_edge.v1];
    edge.v2 = extrude_vert_indices[orig_edge.v2];
    edge.flag = (ME_EDGEDRAW | ME_EDGERENDER);
  }

  for (const int i : new_polys.index_range()) {
    MPoly &poly = new_polys[i];
    poly.loopstart = orig_loop_size + i * 4;
    poly.totloop = 4;
    poly.mat_nr = 0;
    poly.flag = 0;
  }

  /* TODO: Figure out winding order for new faces. */
  for (const int i : selection.index_range()) {
    MutableSpan<MLoop> poly_loops = new_loops.slice(4 * i, 4);
    const int orig_edge_index = selection[i];
    const MEdge &orig_edge = edges[orig_edge_index];
    const MEdge &duplicate_edge = duplicate_edges[i];
    const int new_vert_index_1 = duplicate_edge.v1 - orig_vert_size;
    const int new_vert_index_2 = duplicate_edge.v2 - orig_vert_size;
    const int orig_vert_index_1 = extrude_vert_orig_indices[new_vert_index_1];
    const int orig_vert_index_2 = extrude_vert_orig_indices[new_vert_index_2];
    const MEdge &extrude_edge_1 = extrude_edges[new_vert_index_1];
    const MEdge &extrude_edge_2 = extrude_edges[new_vert_index_2];

    /* Add the start vertex and edge along the original edge. */
    poly_loops[0].v = orig_edge.v1;
    poly_loops[0].e = orig_edge_index;
    /* Add the other vertex of the original edge and the first extrusion edge. */
    poly_loops[1].v = orig_edge.v2;
    poly_loops[1].e = extrude_edge_range.start() + new_vert_index_2;
    /* The first vertex of the duplicate edge is the extrude edge vertex that isn't used yet. */
    poly_loops[2].v = extrude_edge_1.v1 == orig_edge.v2 ? extrude_edge_2.v1 : extrude_edge_2.v2;
    poly_loops[2].e = duplicate_edge_range.start() + i;
    /* The second vertex of the duplicate edge and the extruded edge on other side. */
    poly_loops[3].v = extrude_edge_2.v1 == orig_edge.v1 ? extrude_edge_1.v1 : extrude_edge_1.v2;
    poly_loops[3].e = extrude_edge_range.start() + new_vert_index_1;
  }

  component.attribute_foreach([&](const AttributeIDRef &id, const AttributeMetaData meta_data) {
    if (meta_data.domain == ATTR_DOMAIN_POINT) {
      OutputAttribute attribute = component.attribute_try_get_for_output(
          id, ATTR_DOMAIN_POINT, meta_data.data_type);

      attribute_math::convert_to_static_type(meta_data.data_type, [&](auto dummy) {
        using T = decltype(dummy);
        MutableSpan<T> data = attribute.as_span().typed<T>();
        MutableSpan<T> new_data = data.slice(extrude_vert_range);

        for (const int i : extrude_vert_orig_indices.index_range()) {
          new_data[i] = data[extrude_vert_orig_indices[i]];
        }
      });

      attribute.save();
    }
    else if (meta_data.domain == ATTR_DOMAIN_EDGE) {
      OutputAttribute attribute = component.attribute_try_get_for_output(
          id, ATTR_DOMAIN_EDGE, meta_data.data_type);

      attribute_math::convert_to_static_type(meta_data.data_type, [&](auto dummy) {
        using T = decltype(dummy);
        MutableSpan<T> data = attribute.as_span().typed<T>();
        MutableSpan<T> duplicate_data = data.slice(duplicate_edge_range);

        for (const int i : selection.index_range()) {
          duplicate_data[i] = data[selection[i]];
        }
      });

      attribute.save();
    }
    return true;
  });

  threading::parallel_for(new_verts.index_range(), 1024, [&](const IndexRange range) {
    for (const int i : range) {
      const float3 offset = offsets[extrude_vert_orig_indices[i]];
      add_v3_v3(new_verts[i].co, offset);
    }
  });

  if (attribute_outputs.top_id) {
    save_selection_as_attribute(
        component, attribute_outputs.top_id.get(), ATTR_DOMAIN_EDGE, duplicate_edge_range);
  }
  if (attribute_outputs.side_id) {
    save_selection_as_attribute(component,
                                attribute_outputs.side_id.get(),
                                ATTR_DOMAIN_FACE,
                                IndexRange(orig_polys.size(), new_poly_size));
  }

  BKE_mesh_runtime_clear_cache(&mesh);
  BKE_mesh_normals_tag_dirty(&mesh);

  BKE_mesh_calc_normals(component.get_for_write());
  BLI_assert(BKE_mesh_is_valid(component.get_for_write()));
}

static IndexMask index_mask_from_selection(const VArray<bool> &selection,
                                           Vector<int64_t> &r_indices)
{
  if (selection.is_single()) {
    if (selection.get_internal_single()) {
      return IndexMask(selection.size());
    }
    return IndexMask(0);
  }

  if (selection.is_span()) {
    Span<bool> span = selection.get_internal_span();
    for (const int i : span.index_range()) {
      if (span[i]) {
        r_indices.append(i);
      }
    }
  }
  else {
    for (const int i : selection.index_range()) {
      if (selection[i]) {
        r_indices.append(i);
      }
    }
  }

  return IndexMask(r_indices);
}

// static Array<Vector<int>> polys_of_vert_map()
// {
// }

static void extrude_mesh_faces(MeshComponent &component,
                               const Field<bool> &selection_field,
                               const Field<float3> &offset_field,
                               const AttributeOutputs &attribute_outputs)
{
  Mesh &mesh = *component.get_for_write();
  const int orig_vert_size = mesh.totvert;
  const int orig_edge_size = mesh.totedge;
  Span<MPoly> orig_polys{mesh.mpoly, mesh.totpoly};
  Span<MLoop> orig_loops{mesh.mloop, mesh.totloop};

  GeometryComponentFieldContext poly_context{component, ATTR_DOMAIN_FACE};
  FieldEvaluator poly_evaluator{poly_context, mesh.totpoly};
  poly_evaluator.add(selection_field);
  poly_evaluator.evaluate();
  const VArray<bool> &poly_selection_varray = poly_evaluator.get_evaluated<bool>(0);
  const IndexMask poly_selection = poly_evaluator.get_evaluated_as_mask(0);

  Vector<int64_t> int_selection_indices;
  const VArray<bool> point_selection_varray = component.attribute_try_adapt_domain(
      poly_selection_varray, ATTR_DOMAIN_FACE, ATTR_DOMAIN_POINT);
  const IndexMask point_selection = index_mask_from_selection(point_selection_varray,
                                                              int_selection_indices);

  Array<float3> offsets(orig_vert_size);
  GeometryComponentFieldContext point_context{component, ATTR_DOMAIN_POINT};
  FieldEvaluator point_evaluator{point_context, &point_selection};
  point_evaluator.add_with_destination(offset_field, offsets.as_mutable_span());
  point_evaluator.evaluate();

  /* TODO: See if this section can be simplified by having a precalculated topology map method for
   * retrieving the faces connected to each edge. */

  /* Keep track of the selected face that each edge corresponds to. Only edges with one selected
   * face will have a single associated face. However, we need to keep track of a value for every
   * face in the mesh at this point, because we don't know how many edges will be selected for
   * extrusion in the end. */
  Array<int> edge_face_indices(orig_edge_size, -1);
  Array<int> edge_neighbor_count(orig_edge_size, 0);
  for (const int i_poly : poly_selection) {
    const MPoly &poly = orig_polys[i_poly];
    for (const MLoop &loop : orig_loops.slice(poly.loopstart, poly.totloop)) {
      edge_neighbor_count[loop.e]++;
      edge_face_indices[loop.e] = i_poly;
    }
  }

  Vector<int> in_between_edges;
  /* The extruded face corresponding to each extruded edge. */
  Vector<int> edge_orig_face_indices;
  Vector<int64_t> selected_edges_orig_indices;
  for (const int i_edge : IndexRange(orig_edge_size)) {
    if (edge_neighbor_count[i_edge] == 1) {
      selected_edges_orig_indices.append(i_edge);
      edge_orig_face_indices.append(edge_face_indices[i_edge]);
    }
    else if (edge_neighbor_count[i_edge] > 1) {
      in_between_edges.append(i_edge);
    }
  }
  const IndexMask edge_selection{selected_edges_orig_indices}; /* TODO: Remove. */

  /* Indices into the `duplicate_edges` span for each original selected edge. */
  Array<int> duplicate_edge_indices(orig_edge_size, -1);
  for (const int i : edge_selection.index_range()) {
    duplicate_edge_indices[edge_selection[i]] = i;
  }

  /* Maps vertex indices in the original mesh to the corresponding extruded vertices. */
  Array<int> new_vert_indices(mesh.totvert, -1);
  /* Maps from the index in the added vertices to the original vertex they were newed from. */
  Vector<int> new_vert_orig_indices;
  new_vert_orig_indices.reserve(edge_selection.size());
  for (const int i_edge : edge_selection) {
    const MEdge &edge = mesh.medge[i_edge];

    if (new_vert_indices[edge.v1] == -1) {
      new_vert_indices[edge.v1] = orig_vert_size + new_vert_orig_indices.size();
      new_vert_orig_indices.append(edge.v1);
    }

    if (new_vert_indices[edge.v2] == -1) {
      new_vert_indices[edge.v2] = orig_vert_size + new_vert_orig_indices.size();
      new_vert_orig_indices.append(edge.v2);
    }
  }

  const IndexRange new_vert_range{orig_vert_size, new_vert_orig_indices.size()};
  /* One edge connects each selected vertex to a new vertex on the extruded polygons. */
  const IndexRange connect_edge_range{orig_edge_size, new_vert_range.size()};
  /* Each selected edge is duplicated to form a single edge on the extrusion. */
  const IndexRange duplicate_edge_range{connect_edge_range.one_after_last(),
                                        edge_selection.size()};
  /* Each edge selected for extrusion is extruded into a single face. */
  const IndexRange side_poly_range{orig_polys.size(), edge_selection.size()};
  const IndexRange side_loop_range{orig_loops.size(), side_poly_range.size() * 4};

  expand_mesh_size(mesh,
                   new_vert_range.size(),
                   connect_edge_range.size() + duplicate_edge_range.size(),
                   side_poly_range.size(),
                   side_loop_range.size());

  MutableSpan<MVert> verts{mesh.mvert, mesh.totvert};
  MutableSpan<MVert> new_verts = verts.slice(new_vert_range);
  MutableSpan<MEdge> edges{mesh.medge, mesh.totedge};
  MutableSpan<MEdge> connect_edges = edges.slice(connect_edge_range);
  MutableSpan<MEdge> duplicate_edges = edges.slice(duplicate_edge_range);
  MutableSpan<MPoly> polys{mesh.mpoly, mesh.totpoly};
  MutableSpan<MPoly> new_polys = polys.slice(side_poly_range);
  MutableSpan<MLoop> loops{mesh.mloop, mesh.totloop};
  MutableSpan<MLoop> new_loops = loops.slice(side_loop_range);

  for (MVert &vert : new_verts) {
    vert.flag = 0;
  }

  for (const int i : connect_edges.index_range()) {
    MEdge &edge = connect_edges[i];
    edge.v1 = new_vert_orig_indices[i];
    edge.v2 = orig_vert_size + i;
    edge.flag = (ME_EDGEDRAW | ME_EDGERENDER);
  }

  for (const int i : duplicate_edges.index_range()) {
    const MEdge &orig_edge = edges[edge_selection[i]];
    MEdge &edge = duplicate_edges[i];
    edge.v1 = new_vert_indices[orig_edge.v1];
    edge.v2 = new_vert_indices[orig_edge.v2];
    edge.flag = (ME_EDGEDRAW | ME_EDGERENDER);
  }

  for (const int i : new_polys.index_range()) {
    MPoly &poly = new_polys[i];
    poly.loopstart = side_loop_range.start() + i * 4;
    poly.totloop = 4;
    poly.mat_nr = 0;
    poly.flag = 0;
  }

  /* Maps new vertices to the extruded edges connecting them to the original edges. The values are
   * indices into the `extrude_edges` array, and the element index corresponds to the vert in
   * `new_verts` of the same index. */
  Array<int> new_vert_to_connect_edge(new_vert_range.size());
  for (const int i : connect_edges.index_range()) {
    const MEdge &connect_edge = connect_edges[i];
    BLI_assert(connect_edge.v1 >= orig_vert_size || connect_edge.v2 >= orig_vert_size);
    const int vert_index = connect_edge.v1 > orig_vert_size ? connect_edge.v1 : connect_edge.v2;
    new_vert_to_connect_edge[vert_index - orig_vert_size] = i;
  }

  /* TODO: Figure out winding order for new faces. */
  for (const int i : edge_selection.index_range()) {
    MutableSpan<MLoop> poly_loops = new_loops.slice(4 * i, 4);
    const int orig_edge_index = edge_selection[i];
    const MEdge &orig_edge = edges[orig_edge_index];
    const MEdge &duplicate_edge = duplicate_edges[i];
    const int new_vert_index_1 = duplicate_edge.v1 - orig_vert_size;
    const int new_vert_index_2 = duplicate_edge.v2 - orig_vert_size;
    const int connect_edge_index_1 = new_vert_to_connect_edge[new_vert_index_1];
    const int connect_edge_index_2 = new_vert_to_connect_edge[new_vert_index_2];
    const MEdge &connect_edge_1 = connect_edges[new_vert_to_connect_edge[new_vert_index_1]];
    const MEdge &connect_edge_2 = connect_edges[new_vert_to_connect_edge[new_vert_index_2]];
    poly_loops[0].v = orig_edge.v1;
    poly_loops[0].e = orig_edge_index;
    poly_loops[1].v = orig_edge.v2;
    poly_loops[1].e = connect_edge_range.start() + connect_edge_index_2;
    /* The first vertex of the duplicate edge is the connect edge that isn't used yet. */
    poly_loops[2].v = connect_edge_1.v1 == orig_edge.v2 ? connect_edge_2.v1 : connect_edge_2.v2;
    poly_loops[2].e = duplicate_edge_range.start() + i;

    poly_loops[3].v = connect_edge_2.v1 == orig_edge.v1 ? connect_edge_1.v1 : connect_edge_1.v2;
    poly_loops[3].e = connect_edge_range.start() + connect_edge_index_1;
  }

  /* Connect original edges that are in between two selected faces to the new vertices.
   * The edge's vertices may have been extruded even though the edge itself was not. */
  for (const int i : in_between_edges) {
    MEdge &edge = edges[i];
    if (new_vert_indices[edge.v1] != -1) {
      edge.v1 = new_vert_indices[edge.v1];
    }
    if (new_vert_indices[edge.v2] != -1) {
      edge.v2 = new_vert_indices[edge.v2];
    }
  }

  /* Connect the selected faces to the extruded/"duplicated" edges and the new vertices. */
  for (const int i_poly : poly_selection) {
    const MPoly &poly = polys[i_poly];
    for (MLoop &loop : loops.slice(poly.loopstart, poly.totloop)) {
      if (new_vert_indices[loop.v] != -1) {
        loop.v = new_vert_indices[loop.v];
      }
      if (duplicate_edge_indices[loop.e] != -1) {
        loop.e = duplicate_edge_range.start() + duplicate_edge_indices[loop.e];
      }
    }
  }

  component.attribute_foreach([&](const AttributeIDRef &id, const AttributeMetaData meta_data) {
    OutputAttribute attribute = component.attribute_try_get_for_output(
        id, meta_data.domain, meta_data.data_type);
    if (!attribute) {
      return true; /* Impossible to write the "normal" attribute. */
    }

    attribute_math::convert_to_static_type(meta_data.data_type, [&](auto dummy) {
      using T = decltype(dummy);
      MutableSpan<T> data = attribute.as_span().typed<T>();
      switch (attribute.domain()) {
        case ATTR_DOMAIN_POINT: {
          MutableSpan<T> new_data = data.slice(new_vert_range);
          for (const int i : new_vert_orig_indices.index_range()) {
            new_data[i] = data[new_vert_orig_indices[i]];
          }
          break;
        }
        case ATTR_DOMAIN_EDGE: {
          MutableSpan<T> duplicate_data = data.slice(duplicate_edge_range);
          MutableSpan<T> connect_data = data.slice(connect_edge_range);
          connect_data.fill(T());
          for (const int i : edge_selection.index_range()) {
            duplicate_data[i] = data[edge_selection[i]];
          }
          break;
        }
        case ATTR_DOMAIN_FACE: {
          MutableSpan<T> new_data = data.slice(side_poly_range);
          for (const int i : new_data.index_range()) {
            new_data[i] = data[edge_orig_face_indices[i]];
          }
          break;
        }
        case ATTR_DOMAIN_CORNER: {
          MutableSpan<T> new_data = data.slice(side_loop_range);
          new_data.fill(T());
          break;
        }
        default:
          BLI_assert_unreachable();
      }
    });

    attribute.save();
    return true;
  });

  threading::parallel_for(point_selection.index_range(), 1024, [&](const IndexRange range) {
    for (const int i : range) {
      const int orig_vert_index = point_selection[i];
      const int new_vert_index = new_vert_indices[orig_vert_index];
      const float3 offset = offsets[orig_vert_index];
      /* If the vertex is used by a selected edge, it will have been duplicated and only the new
       * vertex should use the offset. Otherwise the vertex might still need an offset, but it was
       * reused on the inside of a set of extruded faces. */
      add_v3_v3(verts[(new_vert_index != -1) ? new_vert_index : orig_vert_index].co, offset);
    }
  });

  BKE_mesh_runtime_clear_cache(&mesh);
  BKE_mesh_normals_tag_dirty(&mesh);

  if (attribute_outputs.top_id) {
    save_selection_as_attribute(
        component, attribute_outputs.top_id.get(), ATTR_DOMAIN_FACE, poly_selection);
  }
  if (attribute_outputs.side_id) {
    save_selection_as_attribute(
        component, attribute_outputs.side_id.get(), ATTR_DOMAIN_FACE, side_poly_range);
  }

  BKE_mesh_calc_normals(component.get_for_write());
  BLI_assert(BKE_mesh_is_valid(component.get_for_write()));
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh");
  Field<bool> selection = params.extract_input<Field<bool>>("Selection");
  Field<float3> offset = params.extract_input<Field<float3>>("Offset");
  const NodeGeometryExtrudeMesh &storage = node_storage(params.node());
  GeometryNodeExtrudeMeshMode mode = static_cast<GeometryNodeExtrudeMeshMode>(storage.mode);

  AttributeOutputs attribute_outputs;
  if (params.output_is_required("Top")) {
    attribute_outputs.top_id = StrongAnonymousAttributeID("Top");
  }
  if (params.output_is_required("Side")) {
    attribute_outputs.side_id = StrongAnonymousAttributeID("Side");
  }

  geometry_set.modify_geometry_sets([&](GeometrySet &geometry_set) {
    if (geometry_set.has_mesh()) {
      MeshComponent &component = geometry_set.get_component_for_write<MeshComponent>();
      switch (mode) {
        case GEO_NODE_EXTRUDE_MESH_VERTICES:
          extrude_mesh_vertices(component, selection, offset, attribute_outputs);
          break;
        case GEO_NODE_EXTRUDE_MESH_EDGES:
          extrude_mesh_edges(component, selection, offset, attribute_outputs);
          break;
        case GEO_NODE_EXTRUDE_MESH_FACES: {
          extrude_mesh_faces(component, selection, offset, attribute_outputs);
          break;
        }
      }
    }
  });

  params.set_output("Mesh", std::move(geometry_set));
  if (attribute_outputs.top_id) {
    params.set_output("Top",
                      AnonymousAttributeFieldInput::Create<bool>(
                          std::move(attribute_outputs.top_id), params.attribute_producer_name()));
  }
  if (attribute_outputs.side_id) {
    params.set_output("Side",
                      AnonymousAttributeFieldInput::Create<bool>(
                          std::move(attribute_outputs.side_id), params.attribute_producer_name()));
  }
}

}  // namespace blender::nodes::node_geo_extrude_mesh_cc

void register_node_type_geo_extrude_mesh()
{
  namespace file_ns = blender::nodes::node_geo_extrude_mesh_cc;

  static bNodeType ntype;
  geo_node_type_base(&ntype, GEO_NODE_EXTRUDE_MESH, "Extrude Mesh", NODE_CLASS_GEOMETRY, 0);
  ntype.declare = file_ns::node_declare;
  node_type_init(&ntype, file_ns::node_init);
  ntype.geometry_node_execute = file_ns::node_geo_exec;
  node_type_storage(
      &ntype, "NodeGeometryExtrudeMesh", node_free_standard_storage, node_copy_standard_storage);
  ntype.draw_buttons = file_ns::node_layout;
  nodeRegisterType(&ntype);
}
