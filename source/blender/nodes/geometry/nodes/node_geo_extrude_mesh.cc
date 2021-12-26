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

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Mesh").supported_type(GEO_COMPONENT_TYPE_MESH);
  b.add_input<decl::Bool>(N_("Selection")).default_value(true).supports_field().hide_value();
  b.add_input<decl::Vector>(N_("Offset")).supports_field().subtype(PROP_TRANSLATION);
  b.add_output<decl::Geometry>("Mesh");
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

static void extrude_mesh_vertices(MeshComponent &component,
                                  const Field<bool> &selection_field,
                                  const Field<float3> &offset_field)
{
  Mesh &mesh = *component.get_for_write();

  GeometryComponentFieldContext context{component, ATTR_DOMAIN_POINT};
  FieldEvaluator evaluator{context, mesh.totvert};
  evaluator.add(offset_field);
  evaluator.set_selection(selection_field);
  evaluator.evaluate();
  const IndexMask selection = evaluator.get_evaluated_selection_as_mask();
  const VArray<float3> offsets = evaluator.get_evaluated<float3>(0);

  const int orig_vert_size = mesh.totvert;
  mesh.totvert += selection.size();
  mesh.totedge += selection.size();

  /* TODO: This is a stupid way to work around an issue with #CustomData_realloc,
   * which doesn't reallocate a referenced layer apparently. */
  CustomData_duplicate_referenced_layers(&mesh.vdata, mesh.totvert);
  CustomData_duplicate_referenced_layers(&mesh.edata, mesh.totedge);

  CustomData_realloc(&mesh.vdata, mesh.totvert);
  CustomData_realloc(&mesh.edata, mesh.totedge);
  BKE_mesh_update_customdata_pointers(&mesh, false);

  MutableSpan<MVert> verts{mesh.mvert, mesh.totvert};
  MutableSpan<MEdge> edges{mesh.medge, mesh.totedge};
  MutableSpan<MVert> new_verts = verts.take_back(selection.size());
  MutableSpan<MEdge> new_edges = edges.take_back(selection.size());

  for (const int i : selection.index_range()) {
    new_edges[i].v1 = selection[i];
    new_edges[i].v2 = orig_vert_size + i;
    new_edges[i].flag |= ME_LOOSEEDGE;
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

  BKE_mesh_runtime_clear_cache(&mesh);
  BKE_mesh_normals_tag_dirty(&mesh);
}

static void extrude_mesh(MeshComponent &component,
                         GeometryNodeExtrudeMeshMode mode,
                         const Field<bool> &selection,
                         const Field<float3> &offset)
{
  switch (mode) {
    case GEO_NODE_EXTRUDE_MESH_VERTICES:
      extrude_mesh_vertices(component, selection, offset);
      break;
    case GEO_NODE_EXTRUDE_MESH_EDGES:
      break;
    case GEO_NODE_EXTRUDE_MESH_FACES:
      break;
  }
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh");
  Field<bool> selection = params.extract_input<Field<bool>>("Selection");
  Field<float3> offset = params.extract_input<Field<float3>>("Offset");
  const NodeGeometryExtrudeMesh &storage = node_storage(params.node());
  GeometryNodeExtrudeMeshMode mode = static_cast<GeometryNodeExtrudeMeshMode>(storage.mode);
  geometry_set.modify_geometry_sets([&](GeometrySet &geometry_set) {
    if (geometry_set.has_mesh()) {
      MeshComponent &component = geometry_set.get_component_for_write<MeshComponent>();
      extrude_mesh(component, mode, selection, offset);
    }
  });
  params.set_output("Mesh", std::move(geometry_set));
}

}  // namespace blender::nodes::node_geo_extrude_mesh_cc

void register_node_type_geo_extrude_mesh()
{
  namespace file_ns = blender::nodes::node_geo_extrude_mesh_cc;

  static bNodeType ntype;
  geo_node_type_base(&ntype, GEO_NODE_EXTRUDE_MESH, "Extrude Mesh", NODE_CLASS_GEOMETRY, 0);
  node_type_init(&ntype, file_ns::node_init);
  ntype.declare = file_ns::node_declare;
  ntype.geometry_node_execute = file_ns::node_geo_exec;
  node_type_storage(
      &ntype, "NodeGeometryExtrudeMesh", node_free_standard_storage, node_copy_standard_storage);
  ntype.draw_buttons = file_ns::node_layout;
  nodeRegisterType(&ntype);
}
