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

#include "UI_interface.h"
#include "UI_resources.h"

#include "BKE_attribute_math.hh"
#include "BKE_customdata.h"
#include "BKE_spline.hh"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_evaluate_curve_in[] = {
    {SOCK_GEOMETRY, N_("Curve")},
    {SOCK_FLOAT,
     N_("Length"),
     0.0f,
     0.0f,
     0.0f,
     0.0f,
     -FLT_MAX,
     FLT_MAX,
     PROP_TRANSLATION,
     SOCK_FIELD},
    {SOCK_RGBA, N_("Custom"), 1, 1, 1, 1, 0, 1, PROP_NONE, SOCK_FIELD},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_evaluate_curve_out[] = {
    {SOCK_VECTOR, N_("Position")},
    {SOCK_VECTOR, N_("Tangent")},
    {SOCK_VECTOR, N_("Normal")},
    {SOCK_RGBA, N_("Custom")},
    {-1, ""},
};

namespace blender::nodes {

class EvaluateCurveFunction : public fn::MultiFunction {
 private:
  GeometrySet geometry_set_;
  AnonymousCustomDataLayerID *attribute_id_;

 public:
  EvaluateCurveFunction(GeometrySet geometry_set, AnonymousCustomDataLayerID *attribute_id)
      : geometry_set_(std::move(geometry_set)), attribute_id_(attribute_id)
  {
    static fn::MFSignature signature = create_signature();
    this->set_signature(&signature);
    CustomData_anonymous_id_strong_increment(attribute_id_);
  }

  ~EvaluateCurveFunction() override
  {
    CustomData_anonymous_id_strong_decrement(attribute_id_);
  }

  static fn::MFSignature create_signature()
  {
    blender::fn::MFSignatureBuilder signature{"Evaluate Curve"};
    signature.single_input<float>("Length");
    signature.single_output<float3>("Position");
    signature.single_output<float3>("Tangent");
    signature.single_output<float3>("Normal");
    signature.single_output<ColorGeometry4f>("Custom");
    return signature.build();
  }

  void call(IndexMask mask, fn::MFParams params, fn::MFContext UNUSED(context)) const override
  {
    const VArray<float> &src_lengths = params.readonly_single_input<float>(0, "Length");
    MutableSpan<float3> sampled_positions = params.uninitialized_single_output<float3>(1,
                                                                                       "Position");
    MutableSpan<float3> sampled_tangents = params.uninitialized_single_output<float3>(2,
                                                                                      "Tangent");
    MutableSpan<float3> sampled_normals = params.uninitialized_single_output<float3>(3, "Normal");
    MutableSpan<ColorGeometry4f> sampled_custom =
        params.uninitialized_single_output<ColorGeometry4f>(4, "Custom");

    auto return_default = [&]() {
      sampled_positions.fill_indices(mask, {0, 0, 0});
      sampled_tangents.fill_indices(mask, {0, 0, 0});
      sampled_normals.fill_indices(mask, {0, 0, 0});
      sampled_custom.fill_indices(mask, {0, 0, 0, 1});
    };

    if (!geometry_set_.has_curve()) {
      return return_default();
    }

    const CurveComponent *curve_component = geometry_set_.get_component_for_read<CurveComponent>();
    const CurveEval *curve = curve_component->get_for_read();
    if (curve->splines().is_empty()) {
      return return_default();
    }
    const Spline &spline = *curve->splines()[0];
    std::optional<GSpan> custom_generic = spline.attributes.get_anonymous_for_read(*attribute_id_);
    if (!custom_generic) {
      return return_default();
    }

    Span<ColorGeometry4f> custom = (*custom_generic).typed<ColorGeometry4f>();
    GVArray_Typed<ColorGeometry4f> evaluated_custom = spline.interpolate_to_evaluated(custom);

    const float spline_length = spline.length();
    const Span<float3> evaluated_positions = spline.evaluated_positions();
    const Span<float3> evaluated_tangents = spline.evaluated_tangents();
    const Span<float3> evaluated_normals = spline.evaluated_normals();
    for (const int i : mask) {
      const float length = std::clamp(src_lengths[i], 0.0f, spline_length);
      Spline::LookupResult lookup = spline.lookup_evaluated_length(length);
      const int i1 = lookup.evaluated_index;
      const int i2 = lookup.next_evaluated_index;
      sampled_positions[i] = attribute_math::mix2(
          lookup.factor, evaluated_positions[i1], evaluated_positions[i2]);
      sampled_tangents[i] = attribute_math::mix2(
                                lookup.factor, evaluated_tangents[i1], evaluated_tangents[i2])
                                .normalized();
      sampled_normals[i] = attribute_math::mix2(
                               lookup.factor, evaluated_normals[i1], evaluated_normals[i2])
                               .normalized();
      sampled_custom[i] = attribute_math::mix2(
          lookup.factor, evaluated_custom[i1], evaluated_custom[i2]);
    }
  }
};

static void geo_node_evaluate_curve_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Curve");
  geometry_set = geometry_set_realize_instances(geometry_set);

  FieldPtr curve_field = params.get_input_field<float>("Length").field();
  bke::FieldRef<ColorGeometry4f> attribute_field = params.get_input_field<ColorGeometry4f>(
      "Custom");

  AnonymousCustomDataLayerID *layer_id = CustomData_anonymous_id_new("Evaluate Curve");
  CurveComponent &curve_component = geometry_set.get_component_for_write<CurveComponent>();
  try_freeze_field_on_geometry(
      curve_component, *layer_id, ATTR_DOMAIN_POINT, *attribute_field.field());

  auto make_output_field = [&](int out_param_index) -> FieldPtr {
    auto fn = std::make_unique<EvaluateCurveFunction>(geometry_set, layer_id);
    return new bke::MultiFunctionField(Vector<FieldPtr>{curve_field},
                                       optional_ptr<const fn::MultiFunction>{std::move(fn)},
                                       out_param_index);
  };

  params.set_output("Position", bke::FieldRef<float3>(make_output_field(1)));
  params.set_output("Tangent", bke::FieldRef<float3>(make_output_field(2)));
  params.set_output("Normal", bke::FieldRef<float3>(make_output_field(3)));
  params.set_output("Custom", bke::FieldRef<ColorGeometry4f>(make_output_field(4)));
}

}  // namespace blender::nodes

void register_node_type_geo_evaluate_curve()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_EVALUATE_CURVE, "Evaluate Curve", NODE_CLASS_ATTRIBUTE, 0);
  node_type_socket_templates(&ntype, geo_node_evaluate_curve_in, geo_node_evaluate_curve_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_evaluate_curve_exec;
  nodeRegisterType(&ntype);
}
