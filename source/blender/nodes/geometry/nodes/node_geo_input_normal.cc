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

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_mesh.h"

#include "node_geometry_util.hh"

namespace blender::nodes {

static void geo_node_input_normal_declare(NodeDeclarationBuilder &b)
{
  b.add_output<decl::Vector>("Normal");
}

static GVArrayPtr mesh_face_normals_gvarray(const Mesh &mesh)
{
  Span<float3> face_normals{(float3 *)BKE_mesh_ensure_face_normals(&mesh), mesh.totpoly};
  return std::make_unique<fn::GVArray_For_GSpan>(face_normals);
}

static GVArrayPtr mesh_vert_normals_gvarray(const Mesh &mesh)
{
  Span<float3> vert_normals{(float3 *)BKE_mesh_ensure_vertex_normals(&mesh), mesh.totvert};
  return std::make_unique<fn::GVArray_For_GSpan>(vert_normals);
}

static const GVArray *construct_mesh_normals_gvarray(const MeshComponent &mesh_component,
                                                     const Mesh &mesh,
                                                     const IndexMask mask,
                                                     const AttributeDomain domain,
                                                     ResourceScope &scope)
{
  switch (domain) {
    case ATTR_DOMAIN_FACE: {
      return scope.add_value(mesh_face_normals_gvarray(mesh)).get();
    }
    case ATTR_DOMAIN_POINT: {
      return scope.add_value(mesh_vert_normals_gvarray(mesh)).get();
    }
    case ATTR_DOMAIN_EDGE: {
      /* In this case, start with vertex normals and convert to the edge domain, since the
       * conversion from edges to vertices is very simple. Use "manual" domain interpolation
       * instead of the GeometryComponent API to avoid calculating unnecessary values and to
       * allow normalizing the result much more simply. */
      Span<float3> vert_normals{(float3 *)BKE_mesh_ensure_vertex_normals(&mesh), mesh.totvert};
      Array<float3> edge_normals(mask.min_array_size());
      Span<MEdge> edges{mesh.medge, mesh.totedge};
      for (const int i : mask) {
        const MEdge &edge = edges[i];
        edge_normals[i] =
            float3::interpolate(vert_normals[edge.v1], vert_normals[edge.v2], 0.5f).normalized();
      }

      return &scope.construct<fn::GVArray_For_ArrayContainer<Array<float3>>>(
          std::move(edge_normals));
    }
    case ATTR_DOMAIN_CORNER: {
      /* The normals on corners are just the mesh's face normals, so start with the face normal
       * array and copy the face normal for each of its corners. In this case using the mesh
       * component's generic domain interpolation is fine, the data will still be normalized,
       * since the face normal is just copied to every corner. */
      GVArrayPtr loop_normals = mesh_component.attribute_try_adapt_domain(
          mesh_face_normals_gvarray(mesh), ATTR_DOMAIN_FACE, ATTR_DOMAIN_CORNER);
      return scope.add_value(std::move(loop_normals)).get();
    }
    default:
      return nullptr;
  }
}

class NormalFieldInput final : public fn::FieldInput {
 public:
  NormalFieldInput() : fn::FieldInput(CPPType::get<float3>(), "Normal")
  {
  }

  const GVArray *get_varray_for_context(const fn::FieldContext &context,
                                        IndexMask mask,
                                        ResourceScope &scope) const final
  {
    if (const GeometryComponentFieldContext *geometry_context =
            dynamic_cast<const GeometryComponentFieldContext *>(&context)) {

      const GeometryComponent &component = geometry_context->geometry_component();
      const AttributeDomain domain = geometry_context->domain();

      if (component.type() == GEO_COMPONENT_TYPE_MESH) {
        const MeshComponent &mesh_component = static_cast<const MeshComponent &>(component);
        const Mesh *mesh = mesh_component.get_for_read();
        if (mesh == nullptr) {
          return nullptr;
        }

        return construct_mesh_normals_gvarray(mesh_component, *mesh, mask, domain, scope);
      }
      if (component.type() == GEO_COMPONENT_TYPE_CURVE) {
        /* TODO: Add curve normals support. */
        return nullptr;
      }
    }
    return nullptr;
  }

  uint64_t hash() const override
  {
    /* Some random constant hash. */
    return 669605641;
  }

  bool is_equal_to(const fn::FieldNode &other) const override
  {
    return dynamic_cast<const NormalFieldInput *>(&other) != nullptr;
  }
};

static void geo_node_input_normal_exec(GeoNodeExecParams params)
{
  Field<float3> normal_field{std::make_shared<NormalFieldInput>()};
  params.set_output("Normal", std::move(normal_field));
}

}  // namespace blender::nodes

void register_node_type_geo_input_normal()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_INPUT_NORMAL, "Normal", NODE_CLASS_INPUT, 0);
  ntype.geometry_node_execute = blender::nodes::geo_node_input_normal_exec;
  ntype.declare = blender::nodes::geo_node_input_normal_declare;
  nodeRegisterType(&ntype);
}
