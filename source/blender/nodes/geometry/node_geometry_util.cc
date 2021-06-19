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
#include "node_util.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_mesh.h"
#include "BKE_mesh_runtime.h"
#include "BKE_pointcloud.h"

namespace blender::nodes {

using bke::GeometryInstanceGroup;

/**
 * Update the availability of a group of input sockets with the same name,
 * used for switching between attribute inputs or single values.
 *
 * \param mode: Controls which socket of the group to make available.
 * \param name_is_available: If false, make all sockets with this name unavailable.
 */
void update_attribute_input_socket_availabilities(bNode &node,
                                                  const StringRef name,
                                                  const GeometryNodeAttributeInputMode mode,
                                                  const bool name_is_available)
{
  const GeometryNodeAttributeInputMode mode_ = (GeometryNodeAttributeInputMode)mode;
  LISTBASE_FOREACH (bNodeSocket *, socket, &node.inputs) {
    if (name == socket->name) {
      const bool socket_is_available =
          name_is_available &&
          ((socket->type == SOCK_STRING && mode_ == GEO_NODE_ATTRIBUTE_INPUT_ATTRIBUTE) ||
           (socket->type == SOCK_ATTRIBUTE && mode_ == GEO_NODE_ATTRIBUTE_INPUT_ATTRIBUTE) ||
           (socket->type == SOCK_FLOAT && mode_ == GEO_NODE_ATTRIBUTE_INPUT_FLOAT) ||
           (socket->type == SOCK_INT && mode_ == GEO_NODE_ATTRIBUTE_INPUT_INTEGER) ||
           (socket->type == SOCK_VECTOR && mode_ == GEO_NODE_ATTRIBUTE_INPUT_VECTOR) ||
           (socket->type == SOCK_RGBA && mode_ == GEO_NODE_ATTRIBUTE_INPUT_COLOR));
      nodeSetSocketAvailability(socket, socket_is_available);
    }
  }
}

void set_attribute_socket_data_type(bNode &node,
                                    const StringRef name,
                                    eNodeSocketDatatype data_type)
{
  LISTBASE_FOREACH (bNodeSocket *, socket, &node.inputs) {
    if (socket->type == SOCK_ATTRIBUTE && name == socket->name) {
      ((bNodeSocketValueAttribute *)socket->default_value)->data_type = data_type;
    }
  }
  LISTBASE_FOREACH (bNodeSocket *, socket, &node.outputs) {
    if (socket->type == SOCK_ATTRIBUTE && name == socket->name) {
      ((bNodeSocketValueAttribute *)socket->default_value)->data_type = data_type;
    }
  }
}

static eNodeSocketDatatype customdata_to_socket_type(CustomDataType custom_data_type)
{
  switch (custom_data_type) {
    case CD_AUTO_FROM_NAME:
      return SOCK_ATTRIBUTE;
    case CD_PROP_FLOAT:
      return SOCK_FLOAT;
    case CD_PROP_INT32:
      return SOCK_INT;
    case CD_PROP_FLOAT3:
      return SOCK_VECTOR;
    case CD_PROP_COLOR:
      return SOCK_RGBA;
    case CD_MLOOPCOL:
      return SOCK_RGBA;
    case CD_PROP_STRING:
      return SOCK_STRING;
    case CD_PROP_BOOL:
      return SOCK_BOOLEAN;
    case CD_PROP_FLOAT2:
      return SOCK_VECTOR;
    default:
      BLI_assert_unreachable();
      return SOCK_ATTRIBUTE;
  }
  BLI_assert_unreachable();
}

void set_attribute_socket_data_type(bNode &node,
                                    const StringRef name,
                                    CustomDataType custom_data_type)
{
  set_attribute_socket_data_type(node, name, customdata_to_socket_type(custom_data_type));
}

void reset_attribute_socket_data_type(bNode &node, const StringRef name)
{
  set_attribute_socket_data_type(node, name, SOCK_ATTRIBUTE);
}

}  // namespace blender::nodes

bool geo_node_poll_default(bNodeType *UNUSED(ntype),
                           bNodeTree *ntree,
                           const char **r_disabled_hint)
{
  if (!STREQ(ntree->idname, "GeometryNodeTree")) {
    *r_disabled_hint = "Not a geometry node tree";
    return false;
  }
  return true;
}

void geo_node_type_base(bNodeType *ntype, int type, const char *name, short nclass, short flag)
{
  node_type_base(ntype, type, name, nclass, flag);
  ntype->poll = geo_node_poll_default;
  ntype->update_internal_links = node_update_internal_links_default;
  ntype->insert_link = node_insert_link_default;
}
