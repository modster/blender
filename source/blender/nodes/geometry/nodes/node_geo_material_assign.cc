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

static bNodeSocketTemplate geo_node_material_assign_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_MATERIAL,
     N_("Material"),
     0.0f,
     0.0f,
     0.0f,
     0.0f,
     0.0f,
     0.0f,
     PROP_NONE,
     SOCK_HIDE_LABEL},
    {SOCK_BOOLEAN, N_("Selection"), 1, 0, 0, 0, 0, 0, PROP_NONE, SOCK_HIDE_VALUE | SOCK_IS_FIELD},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_material_assign_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

static void assign_material_to_faces(Mesh &mesh, const VArray<bool> &face_mask, Material *material)
{
  int new_material_index = -1;
  for (const int i : IndexRange(mesh.totcol)) {
    Material *other_material = mesh.mat[i];
    if (other_material == material) {
      new_material_index = i;
      break;
    }
  }
  if (new_material_index == -1) {
    /* Append a new material index. */
    new_material_index = mesh.totcol;
    BKE_id_material_eval_assign(&mesh.id, new_material_index + 1, material);
  }

  mesh.mpoly = (MPoly *)CustomData_duplicate_referenced_layer(&mesh.pdata, CD_MPOLY, mesh.totpoly);
  for (const int i : IndexRange(mesh.totpoly)) {
    if (face_mask[i]) {
      MPoly &poly = mesh.mpoly[i];
      poly.mat_nr = new_material_index;
    }
  }
}

static void geo_node_material_assign_exec(GeoNodeExecParams params)
{
  Material *material = params.extract_input<Material *>("Material");
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  geometry_set = geometry_set_realize_instances(geometry_set);

  if (geometry_set.has<MeshComponent>()) {
    MeshComponent &mesh_component = geometry_set.get_component_for_write<MeshComponent>();
    Mesh *mesh = mesh_component.get_for_write();
    if (mesh != nullptr) {

      bke::FieldRef<bool> field = params.get_input_field<bool>("Selection");
      bke::FieldInputs field_inputs = field->prepare_inputs();
      Vector<std::unique_ptr<bke::FieldInputValue>> field_input_values;
      prepare_field_inputs(field_inputs, mesh_component, ATTR_DOMAIN_FACE, field_input_values);
      bke::FieldOutput field_output = field->evaluate(
          IndexRange(mesh_component.attribute_domain_size(ATTR_DOMAIN_FACE)), field_inputs);

      GVArray_Typed<bool> selection{field_output.varray_ref()};

      assign_material_to_faces(*mesh, *selection, material);
    }
  }

  params.set_output("Geometry", std::move(geometry_set));
}

}  // namespace blender::nodes

void register_node_type_geo_material_assign()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_MATERIAL_ASSIGN, "Material Assign", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_material_assign_in, geo_node_material_assign_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_material_assign_exec;
  nodeRegisterType(&ntype);
}
