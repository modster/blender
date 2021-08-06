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

#include "node_geometry_util.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_material.h"

static bNodeSocketTemplate geo_node_geometry_expander_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

static void geo_node_geometry_expander_layout(uiLayout *layout,
                                              bContext *UNUSED(C),
                                              PointerRNA *UNUSED(ptr))
{
  uiItemO(layout, "Add", ICON_ADD, "node.geometry_expander_output_add");
}

static bool geo_node_geometry_expander_socket_layout(const bContext *UNUSED(C),
                                                     uiLayout *layout,
                                                     bNodeTree *ntree,
                                                     bNode *node,
                                                     bNodeSocket *socket)
{
  if (socket->in_out == SOCK_IN) {
    return false;
  }

  const NodeGeometryGeometryExpander *storage = (const NodeGeometryGeometryExpander *)
                                                    node->storage;
  const int socket_index = BLI_findindex(&node->outputs, socket);

  GeometryExpanderOutput *expander_output = (GeometryExpanderOutput *)BLI_findlink(
      &storage->outputs, socket_index);
  nodeGeometryExpanderUpdateOutputNameCache(expander_output, ntree);

  PointerRNA expander_output_ptr;
  RNA_pointer_create(
      &ntree->id, &RNA_GeometryExpanderOutput, expander_output, &expander_output_ptr);

  uiLayout *row = uiLayoutRow(layout, true);
  uiLayout *split = uiLayoutSplit(row, 0.7, false);
  uiItemL(split, expander_output->display_name_cache, ICON_NONE);
  uiLayout *subrow = uiLayoutRow(split, true);
  if (expander_output->is_outdated) {
    uiItemL(subrow, "", ICON_ERROR);
  }
  else {
    uiItemR(subrow, &expander_output_ptr, "array_source", 0, "", ICON_NONE);
  }
  uiItemIntO(
      subrow, "", ICON_X, "node.geometry_expander_output_remove", "output_index", socket_index);

  return true;
}

static GeometryComponentType array_source_to_component_type(
    const eGeometryExpanderArraySource array_source)
{
  switch (array_source) {
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_MESH_VERTICES:
      return GEO_COMPONENT_TYPE_MESH;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_MESH_EDGES:
      return GEO_COMPONENT_TYPE_MESH;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_MESH_FACES:
      return GEO_COMPONENT_TYPE_MESH;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_MESH_FACE_CORNERS:
      return GEO_COMPONENT_TYPE_MESH;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_POINT_CLOUD_POINTS:
      return GEO_COMPONENT_TYPE_POINT_CLOUD;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_CURVE_POINTS:
      return GEO_COMPONENT_TYPE_CURVE;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_CURVE_SPLINES:
      return GEO_COMPONENT_TYPE_CURVE;
  }
  BLI_assert_unreachable();
  return GEO_COMPONENT_TYPE_MESH;
}

static AttributeDomain array_source_to_domain(const eGeometryExpanderArraySource array_source)
{
  switch (array_source) {
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_MESH_VERTICES:
      return ATTR_DOMAIN_POINT;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_MESH_EDGES:
      return ATTR_DOMAIN_EDGE;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_MESH_FACES:
      return ATTR_DOMAIN_FACE;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_MESH_FACE_CORNERS:
      return ATTR_DOMAIN_CORNER;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_POINT_CLOUD_POINTS:
      return ATTR_DOMAIN_POINT;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_CURVE_POINTS:
      return ATTR_DOMAIN_POINT;
    case GEOMETRY_EXPANDER_ARRAY_SOURCE_CURVE_SPLINES:
      return ATTR_DOMAIN_CURVE;
  }
  BLI_assert_unreachable();
  return ATTR_DOMAIN_POINT;
}

static void geo_node_geometry_expander_exec(GeoNodeExecParams params)
{
  const bNode &bnode = params.node();
  const bNodeTree &ntree = params.ntree();
  const NodeGeometryGeometryExpander *storage = (const NodeGeometryGeometryExpander *)
                                                    bnode.storage;
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  int socket_index;
  LISTBASE_FOREACH_INDEX (
      GeometryExpanderOutput *, expander_output, &storage->outputs, socket_index) {
    bNodeSocket &socket = *(bNodeSocket *)BLI_findlink(&bnode.outputs, socket_index);
    const ArrayCPPType *array_cpp_type = dynamic_cast<const ArrayCPPType *>(
        socket.typeinfo->get_geometry_nodes_cpp_type());
    BLI_assert(array_cpp_type != nullptr);
    const CustomDataType data_type = bke::cpp_type_to_custom_data_type(
        array_cpp_type->element_type());
    BUFFER_FOR_CPP_TYPE_VALUE(*array_cpp_type, buffer);
    const eGeometryExpanderArraySource array_source = (eGeometryExpanderArraySource)
                                                          expander_output->array_source;
    const GeometryComponentType component_type = array_source_to_component_type(array_source);
    const AttributeDomain domain = array_source_to_domain(array_source);

    const GeometryComponent *component = geometry_set.get_component_for_read(component_type);
    if (component == nullptr) {
      array_cpp_type->default_construct(buffer);
    }
    else {
      const int domain_size = component->attribute_domain_size(domain);
      switch (expander_output->type) {
        case GEOMETRY_EXPANDER_OUTPUT_TYPE_BUILTIN: {
          GVArrayPtr attribute = component->attribute_try_get_for_read(
              expander_output->builtin_identifier, domain, data_type);
          if (attribute) {
            array_cpp_type->array_construct_uninitialized(buffer, domain_size);
            attribute->materialize_to_uninitialized(array_cpp_type->array_span(buffer).data());
          }
          else {
            array_cpp_type->default_construct(buffer);
          }
          break;
        }
        case GEOMETRY_EXPANDER_OUTPUT_TYPE_INPUT: {
          const std::string attribute_name = params.get_group_input_attribute_name(
              expander_output->input_identifier);
          GVArrayPtr attribute = component->attribute_try_get_for_read(
              attribute_name, domain, data_type);
          if (attribute) {
            array_cpp_type->array_construct_uninitialized(buffer, domain_size);
            attribute->materialize_to_uninitialized(array_cpp_type->array_span(buffer).data());
          }
          else {
            array_cpp_type->default_construct(buffer);
          }
          break;
        }
        case GEOMETRY_EXPANDER_OUTPUT_TYPE_LOCAL: {
          const std::string attribute_name = get_local_attribute_name(
              ntree.id.name,
              expander_output->local_node_name,
              expander_output->local_socket_identifier);
          GVArrayPtr attribute = component->attribute_try_get_for_read(
              attribute_name, domain, data_type);
          if (attribute) {
            array_cpp_type->array_construct_uninitialized(buffer, domain_size);
            attribute->materialize_to_uninitialized(array_cpp_type->array_span(buffer).data());
          }
          else {
            array_cpp_type->default_construct(buffer);
          }
          break;
        }
      }
    }

    params.set_output_by_move(socket.identifier, {array_cpp_type, buffer});
  }
}

static void geo_node_geometry_expander_init(bNodeTree *UNUSED(ntree), bNode *node)
{
  NodeGeometryGeometryExpander *storage = (NodeGeometryGeometryExpander *)MEM_callocN(
      sizeof(NodeGeometryGeometryExpander), __func__);
  node->storage = storage;
}

static void geo_node_geometry_expander_update(bNodeTree *ntree, bNode *node)
{
  NodeGeometryGeometryExpander *storage = (NodeGeometryGeometryExpander *)node->storage;

  Map<StringRef, bNodeSocket *> old_outputs;

  LISTBASE_FOREACH (bNodeSocket *, socket, &node->outputs) {
    old_outputs.add(socket->identifier, socket);
  }
  VectorSet<bNodeSocket *> new_sockets;
  LISTBASE_FOREACH (GeometryExpanderOutput *, expander_output, &storage->outputs) {
    bNodeSocket *socket = old_outputs.lookup_default(expander_output->socket_identifier, nullptr);
    if (socket == nullptr) {
      const char *idname = nodeStaticSocketType(expander_output->socket_type, PROP_NONE);
      socket = nodeAddSocket(
          ntree, node, SOCK_OUT, idname, expander_output->socket_identifier, "name");
    }
    new_sockets.add_new(socket);
  }
  LISTBASE_FOREACH_MUTABLE (bNodeSocket *, socket, &node->outputs) {
    if (!new_sockets.contains(socket)) {
      nodeRemoveSocket(ntree, node, socket);
    }
  }
  BLI_listbase_clear(&node->outputs);
  for (bNodeSocket *socket : new_sockets) {
    BLI_addtail(&node->outputs, socket);
  }
}

static void geo_node_geometry_expander_storage_free(bNode *node)
{
  NodeGeometryGeometryExpander *storage = (NodeGeometryGeometryExpander *)node->storage;
  LISTBASE_FOREACH_MUTABLE (GeometryExpanderOutput *, output, &storage->outputs) {
    MEM_freeN(output);
  }
  MEM_freeN(storage);
}

static void geo_node_geometry_expander_storage_copy(bNodeTree *UNUSED(dest_ntree),
                                                    bNode *dst_node,
                                                    const bNode *src_node)
{
  NodeGeometryGeometryExpander *src_storage = (NodeGeometryGeometryExpander *)src_node->storage;
  NodeGeometryGeometryExpander *dst_storage = (NodeGeometryGeometryExpander *)MEM_callocN(
      sizeof(NodeGeometryGeometryExpander), __func__);
  LISTBASE_FOREACH (GeometryExpanderOutput *, src_output, &src_storage->outputs) {
    GeometryExpanderOutput *dst_output = (GeometryExpanderOutput *)MEM_callocN(
        sizeof(GeometryExpanderOutput), __func__);
    *dst_output = *src_output;
    BLI_addtail(&dst_storage->outputs, dst_output);
  }
  dst_node->storage = dst_storage;
}

}  // namespace blender::nodes

void register_node_type_geo_geometry_expander()
{
  static bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_GEOMETRY_EXPANDER, "Geometry Expander", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_geometry_expander_in, nullptr);
  node_type_init(&ntype, blender::nodes::geo_node_geometry_expander_init);
  node_type_storage(&ntype,
                    "NodeGeometryGeometryExpander",
                    blender::nodes::geo_node_geometry_expander_storage_free,
                    blender::nodes::geo_node_geometry_expander_storage_copy);
  node_type_update(&ntype, blender::nodes::geo_node_geometry_expander_update);
  ntype.geometry_node_execute = blender::nodes::geo_node_geometry_expander_exec;
  ntype.draw_buttons = blender::nodes::geo_node_geometry_expander_layout;
  ntype.draw_socket = blender::nodes::geo_node_geometry_expander_socket_layout;
  nodeRegisterType(&ntype);
}
