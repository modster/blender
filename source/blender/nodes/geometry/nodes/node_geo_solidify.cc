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

#include "BKE_node.h"

#include "DNA_mesh_types.h"
#include "DNA_modifier_types.h"
#include "DNA_node_types.h"

#include "GEO_solidifiy.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_solidify {
static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>(N_("Mesh")).supported_type(GEO_COMPONENT_TYPE_MESH);
  b.add_input<decl::Float>(N_("Thickness"))
      .default_value(0.0f)
      .subtype(PROP_DISTANCE)
      .supports_field();
  b.add_input<decl::Float>(N_("Clamp"))
      .default_value(0.0f)
      .min(0.0f)
      .max(2.0f)
      .subtype(PROP_FACTOR);
  b.add_input<decl::Float>(N_("Offset"))
      .default_value(0.0f)
      .min(-1.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR);
  b.add_input<decl::Bool>(N_("Fill")).default_value(true);
  b.add_input<decl::Bool>(N_("Rim")).default_value(true);

  b.add_output<decl::Geometry>(N_("Mesh"));
  b.add_output<decl::Bool>(N_("Fill Faces")).field_source();
  b.add_output<decl::Bool>(N_("Rim Faces")).field_source();
}

static void node_layout(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
{
  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);
  uiItemR(layout, ptr, "nonmanifold_offset_mode", 0, nullptr, ICON_NONE);
  uiItemR(layout, ptr, "nonmanifold_boundary_mode", 0, nullptr, ICON_NONE);
}

static void geo_node_solidify_init(bNodeTree *UNUSED(tree), bNode *node)
{
  NodeGeometrySolidify *node_storage = (NodeGeometrySolidify *)MEM_callocN(
      sizeof(NodeGeometrySolidify), __func__);

  node->storage = node_storage;
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const bNode &node = params.node();
  NodeGeometrySolidify &node_storage = *(NodeGeometrySolidify *)node.storage;
  const Object *self_object = params.self_object();

  bool add_fill = params.extract_input<bool>("Fill");
  bool add_rim = params.extract_input<bool>("Rim");
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh");

  float offset = params.extract_input<float>("Offset");
  float offset_clamp = params.extract_input<float>("Clamp");

  bke::StrongAnonymousAttributeID fill_id;
  bke::StrongAnonymousAttributeID rim_id;

  if (geometry_set.has<MeshComponent>()) {
    geometry_set.modify_geometry_sets([&](GeometrySet &geometry_set) {
      MeshComponent &component = geometry_set.get_component_for_write<MeshComponent>();
      Mesh *input_mesh = component.get_for_write();

      const int domain_size = component.attribute_domain_size(ATTR_DOMAIN_POINT);
      GeometryComponentFieldContext context{component, ATTR_DOMAIN_POINT};

      Field<float> thickness_field = params.extract_input<Field<float>>("Thickness");
      FieldEvaluator thickness_evaluator{context, domain_size};
      thickness_evaluator.add(thickness_field);
      thickness_evaluator.evaluate();
      Array<float> thickness(domain_size);
      thickness_evaluator.get_evaluated<float>(0).materialize(thickness);

      char flag = 0;

      if (add_fill) {
        flag |= MOD_SOLIDIFY_SHELL;
      }

      if (add_rim) {
        flag |= MOD_SOLIDIFY_RIM;
      }

      SolidifyData solidify_node_data = {
          self_object,
          1,
          offset,
          0.0f,
          offset_clamp,
          node_storage.nonmanifold_offset_mode,
          node_storage.nonmanifold_boundary_mode,
          flag,
          0.01f,
          0.0f,
          thickness.begin(),
      };

      bool *shell_verts = nullptr;
      bool *rim_verts = nullptr;
      bool *shell_faces = nullptr;
      bool *rim_faces = nullptr;

      Mesh *output_mesh = solidify_nonmanifold(
          &solidify_node_data, input_mesh, &shell_verts, &rim_verts, &shell_faces, &rim_faces);

      if (output_mesh != input_mesh) {
        component.replace(output_mesh, GeometryOwnershipType::Editable);

        if (params.output_is_required("Fill Faces")) {
          fill_id = StrongAnonymousAttributeID("fill_faces");
          if (add_fill) {
            OutputAttribute_Typed<bool> shell_faces_attribute =
                component.attribute_try_get_for_output_only<bool>(fill_id.get(), ATTR_DOMAIN_FACE);
            Span<bool> shell_faces_span(shell_faces, shell_faces_attribute->size());
            shell_faces_attribute->set_all(shell_faces_span);
            shell_faces_attribute.save();
          }
        }

        if (params.output_is_required("Rim Faces")) {
          rim_id = StrongAnonymousAttributeID("rim_faces");
          if (add_rim) {
            OutputAttribute_Typed<bool> rim_faces_attribute =
                component.attribute_try_get_for_output_only<bool>(rim_id.get(), ATTR_DOMAIN_FACE);
            Span<bool> rim_faces_span(rim_faces, rim_faces_attribute->size());
            rim_faces_attribute->set_all(rim_faces_span);
            rim_faces_attribute.save();
          }
        }
      }

      MEM_freeN(shell_verts);
      MEM_freeN(rim_verts);
      MEM_freeN(shell_faces);
      MEM_freeN(rim_faces);
    });
  }
  if (fill_id) {
    params.set_output("Fill Faces",
                      AnonymousAttributeFieldInput::Create<bool>(
                          std::move(fill_id), params.attribute_producer_name()));
  }
  if (rim_id) {
    params.set_output("Rim Faces",
                      AnonymousAttributeFieldInput::Create<bool>(
                          std::move(rim_id), params.attribute_producer_name()));
  }

  params.set_output("Mesh", geometry_set);
}

}  // namespace blender::nodes::node_geo_solidify

void register_node_type_geo_solidify()
{
  namespace file_ns = blender::nodes::node_geo_solidify;

  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_SOLIDIFY, "Solidify", NODE_CLASS_GEOMETRY);
  ntype.declare = file_ns::node_declare;
  node_type_storage(
      &ntype, "NodeGeometrySolidify", node_free_standard_storage, node_copy_standard_storage);
  node_type_init(&ntype, file_ns::geo_node_solidify_init);
  node_type_size(&ntype, 172, 100, 600);
  ntype.geometry_node_execute = file_ns::node_geo_exec;
  ntype.draw_buttons = file_ns::node_layout;
  nodeRegisterType(&ntype);
}
