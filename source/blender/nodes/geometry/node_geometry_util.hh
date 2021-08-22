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

#pragma once

#include <string.h>

#include "BLI_float3.hh"
#include "BLI_utildefines.h"

#include "MEM_guardedalloc.h"

#include "DNA_node_types.h"

#include "BKE_node.h"

#include "BLT_translation.h"

#include "NOD_geometry.h"
#include "NOD_geometry_exec.hh"
#include "NOD_runtime_types.hh"

#include "UI_resources.h"

#include "node_util.h"

void geo_node_type_base(
    struct bNodeType *ntype, int type, const char *name, short nclass, short flag);
bool geo_node_poll_default(struct bNodeType *ntype,
                           struct bNodeTree *ntree,
                           const char **r_disabled_hint);

namespace blender::nodes {

namespace detail {
DECL_NODE_FUNC_OPTIONAL(build_multi_function)
DECL_NODE_FUNC_OPTIONAL(geometry_node_execute)
DECL_NODE_FIELD_OPTIONAL(geometry_node_execute_supports_laziness, false);
}  // namespace detail

template<typename T> struct GeometryNodeDefinition : public NodeDefinition<T> {
  static const int ui_icon = ICON_NONE;
  static const short node_class = NODE_CLASS_GEOMETRY;
  inline static const StructRNA *rna_base = &RNA_GeometryNode;

  static bool poll(bNodeType *ntype, bNodeTree *ntree, const char **r_disabled_hint)
  {
    return geo_node_poll_default(ntype, ntree, r_disabled_hint);
  }

  /* Registers a node type using static fields and callbacks of the template argument. */
  static void register_type()
  {
    NodeDefinition<T>::typeinfo_.build_multi_function =
        detail::node_type_get__build_multi_function<T>();
    NodeDefinition<T>::typeinfo_.geometry_node_execute =
        detail::node_type_get__geometry_node_execute<T>();
    NodeDefinition<T>::typeinfo_.geometry_node_execute_supports_laziness =
        detail::node_type_get__geometry_node_execute_supports_laziness<T>();

    NodeDefinition<T>::register_type();
  }
};

void update_attribute_input_socket_availabilities(bNode &node,
                                                  const StringRef name,
                                                  const GeometryNodeAttributeInputMode mode,
                                                  const bool name_is_available = true);

Array<uint32_t> get_geometry_element_ids_as_uints(const GeometryComponent &component,
                                                  const AttributeDomain domain);

void transform_mesh(Mesh *mesh,
                    const float3 translation,
                    const float3 rotation,
                    const float3 scale);

Mesh *create_cylinder_or_cone_mesh(const float radius_top,
                                   const float radius_bottom,
                                   const float depth,
                                   const int verts_num,
                                   const GeometryNodeMeshCircleFillType fill_type);

Mesh *create_cube_mesh(const float size);

/**
 * Copies the point domain attributes from `in_component` that are in the mask to `out_component`.
 */
void copy_point_attributes_based_on_mask(const GeometryComponent &in_component,
                                         GeometryComponent &result_component,
                                         Span<bool> masks,
                                         const bool invert);

struct CurveToPointsResults {
  int result_size;
  MutableSpan<float3> positions;
  MutableSpan<float> radii;
  MutableSpan<float> tilts;

  Map<std::string, GMutableSpan> point_attributes;

  MutableSpan<float3> tangents;
  MutableSpan<float3> normals;
  MutableSpan<float3> rotations;
};
/**
 * Create references for all result point cloud attributes to simplify accessing them later on.
 */
CurveToPointsResults curve_to_points_create_result_attributes(PointCloudComponent &points,
                                                              const CurveEval &curve);

void curve_create_default_rotation_attribute(Span<float3> tangents,
                                             Span<float3> normals,
                                             MutableSpan<float3> rotations);

}  // namespace blender::nodes
