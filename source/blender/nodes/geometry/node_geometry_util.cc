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
#include "BKE_spline.hh"

#include "NOD_type_conversions.hh"

namespace blender::nodes {

using bke::GeometryInstanceGroup;

static void update_multi_type_socket_availabilities(ListBase &socket_list,
                                                    const StringRef name,
                                                    const CustomDataType type,
                                                    const bool name_is_available)
{
  LISTBASE_FOREACH (bNodeSocket *, socket, &socket_list) {
    if (name == socket->name) {
      const bool socket_is_available = name_is_available &&
                                       ((socket->type == SOCK_STRING && type == CD_PROP_STRING) ||
                                        (socket->type == SOCK_FLOAT && type == CD_PROP_FLOAT) ||
                                        (socket->type == SOCK_INT && type == CD_PROP_INT32) ||
                                        (socket->type == SOCK_VECTOR && type == CD_PROP_FLOAT3) ||
                                        (socket->type == SOCK_RGBA && type == CD_PROP_COLOR));
      nodeSetSocketAvailability(socket, socket_is_available);
    }
  }
}

void update_multi_type_input_socket_availabilities(bNode &node,
                                                   const StringRef name,
                                                   const CustomDataType type,
                                                   const bool name_is_available)
{
  update_multi_type_socket_availabilities(node.inputs, name, type, name_is_available);
}

void update_multi_type_output_socket_availabilities(bNode &node,
                                                    const StringRef name,
                                                    const CustomDataType type,
                                                    const bool name_is_available)
{
  update_multi_type_socket_availabilities(node.outputs, name, type, name_is_available);
}

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
           (socket->type == SOCK_FLOAT && mode_ == GEO_NODE_ATTRIBUTE_INPUT_FLOAT) ||
           (socket->type == SOCK_INT && mode_ == GEO_NODE_ATTRIBUTE_INPUT_INTEGER) ||
           (socket->type == SOCK_VECTOR && mode_ == GEO_NODE_ATTRIBUTE_INPUT_VECTOR) ||
           (socket->type == SOCK_RGBA && mode_ == GEO_NODE_ATTRIBUTE_INPUT_COLOR));
      nodeSetSocketAvailability(socket, socket_is_available);
    }
  }
}

void prepare_field_inputs(bke::FieldInputs &field_inputs,
                          const GeometryComponent &component,
                          const AttributeDomain domain,
                          Vector<std::unique_ptr<bke::FieldInputValue>> &r_values)
{
  const int domain_size = component.attribute_domain_size(domain);
  for (const bke::FieldInputKey &key : field_inputs) {
    std::unique_ptr<bke::FieldInputValue> input_value;
    if (const bke::PersistentAttributeFieldInputKey *persistent_attribute_key =
            dynamic_cast<const bke::PersistentAttributeFieldInputKey *>(&key)) {
      const StringRef name = persistent_attribute_key->name();
      const CPPType &cpp_type = persistent_attribute_key->type();
      const CustomDataType type = bke::cpp_type_to_custom_data_type(cpp_type);
      GVArrayPtr attribute = component.attribute_get_for_read(name, domain, type);
      input_value = std::make_unique<bke::GVArrayFieldInputValue>(std::move(attribute));
    }
    else if (dynamic_cast<const bke::IndexFieldInputKey *>(&key) != nullptr) {
      auto index_func = [](int i) { return i; };
      VArrayPtr<int> index_varray = std::make_unique<VArray_For_Func<int, decltype(index_func)>>(
          domain_size, index_func);
      GVArrayPtr index_gvarray = std::make_unique<fn::GVArray_For_VArray<int>>(
          std::move(index_varray));
      input_value = std::make_unique<bke::GVArrayFieldInputValue>(std::move(index_gvarray));
    }
    else if (const bke::AnonymousAttributeFieldInputKey *anonymous_attribute_key =
                 dynamic_cast<const bke::AnonymousAttributeFieldInputKey *>(&key)) {
      const AnonymousCustomDataLayerID &layer_id = anonymous_attribute_key->layer_id();
      ReadAttributeLookup attribute = component.attribute_try_get_anonymous_for_read(layer_id);
      if (!attribute) {
        continue;
      }
      GVArrayPtr varray = std::move(attribute.varray);
      if (attribute.domain != domain) {
        /* TODO: Not all boolean attributes are selections. */
        if (varray->type().is<bool>() && component.type() == GEO_COMPONENT_TYPE_MESH) {
          const MeshComponent &mesh_component = static_cast<const MeshComponent &>(component);
          VArrayPtr<bool> varray_bool = std::make_unique<fn::VArray_For_GVArray<bool>>(
              std::move(varray));
          varray_bool = mesh_component.adapt_selection(
              std::move(varray_bool), attribute.domain, domain);
          if (!varray_bool) {
            continue;
          }
          varray = std::make_unique<fn::GVArray_For_VArray<bool>>(std::move(varray_bool));
        }
        else {
          varray = component.attribute_try_adapt_domain(
              std::move(varray), attribute.domain, domain);
        }
      }
      if (!varray) {
        continue;
      }
      const CPPType &type = anonymous_attribute_key->type();
      if (varray->type() != type) {
        const blender::nodes::DataTypeConversions &conversions = get_implicit_type_conversions();
        varray = conversions.try_convert(std::move(varray), type);
      }
      if (!varray) {
        continue;
      }
      input_value = std::make_unique<bke::GVArrayFieldInputValue>(std::move(varray));
    }
    else if (dynamic_cast<const bke::CurveParameterFieldInputKey *>(&key)) {
      if (component.type() != GEO_COMPONENT_TYPE_CURVE) {
        continue;
      }
      const CurveComponent &curve_component = static_cast<const CurveComponent &>(component);
      const CurveEval *curve = curve_component.get_for_read();
      if (curve == nullptr) {
        continue;
      }

      Span<SplinePtr> splines = curve->splines();
      Array<int> offsets = curve->control_point_offsets();

      Array<float> parameters(offsets.last());

      for (const int i_spline : splines.index_range()) {
        const int offset = offsets[i_spline];
        MutableSpan<float> spline_parameters = parameters.as_mutable_span().slice(
            offset, offsets[i_spline + 1]);
        spline_parameters.first() = 0.0f;

        const Spline &spline = *splines[i_spline];
        const Span<float> lengths_eval = spline.evaluated_lengths();
        const float total_length_inv = spline.length() == 0.0f ? 0.0f : 1.0f / spline.length();
        switch (spline.type()) {
          case Spline::Type::Bezier: {
            const BezierSpline &bezier_spline = static_cast<const BezierSpline &>(spline);
            const Span<int> control_point_offsets = bezier_spline.control_point_offsets();
            for (const int i : IndexRange(1, spline.size() - 1)) {
              spline_parameters[i] = lengths_eval[control_point_offsets[i] - 1];
            }
            break;
          }
          case Spline::Type::Poly: {
            if (spline.is_cyclic()) {
              spline_parameters.drop_front(1).copy_from(lengths_eval.drop_back(1));
            }
            else {
              spline_parameters.drop_front(1).copy_from(lengths_eval);
            }
            break;
          }
          case Spline::Type::NURBS: {
            /* Instead of doing something totally arbirary and wrong for the prototype, just do
             * nothing currently. Consult NURBS experts or something or document this heavily if
             * it ever makes it to master. */
            parameters.as_mutable_span().slice(offset, offsets[i_spline + 1]).fill(0.0f);
            break;
          }
        }

        for (float &parameter : spline_parameters) {
          parameter *= total_length_inv;
        }
      }
      GVArrayPtr varray = std::make_unique<fn::GVArray_For_ArrayContainer<Array<float>>>(
          std::move(parameters));
      input_value = std::make_unique<bke::GVArrayFieldInputValue>(std::move(varray));
    }

    field_inputs.set_input(key, *input_value);
    r_values.append(std::move(input_value));
  }
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
