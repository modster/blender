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

/** \file
 * \ingroup fn
 */

#include "BLI_function_ref.hh"
#include "BLI_map.hh"
#include "BLI_vector.hh"

#include "FN_generic_array.hh"
#include "FN_generic_virtual_array.hh"
#include "FN_multi_function_procedure.hh"
#include "FN_multi_function_procedure_builder.hh"
#include "FN_multi_function_procedure_executor.hh"

namespace blender::fn {

class Field;

class Field {
  fn::CPPType *type_;

 public:
  fn::CPPType &type()
  {
    return *type_;
  }

  virtual void foreach_input_recursive(blender::FunctionRef<void(const InputField &input)> fn) = 0;
  virtual void foreach_input(blender::FunctionRef<void(const InputField &input)> fn) = 0;
};

class InputField : public Field {

 public:
  virtual GVArrayPtr get_data(IndexMask mask) const = 0;
  void foreach_input_recursive(blender::FunctionRef<void(const InputField &input)> fn) final
  {
  }
  void foreach_input(blender::FunctionRef<void(const InputField &input)> fn) final
  {
  }
};

class MultiFunctionField final : public Field {
  blender::Vector<std::shared_ptr<Field>> input_fields_;
  MultiFunction &function_;

 public:
  void foreach_input_recursive(blender::FunctionRef<void(const InputField &input)> fn) final
  {
    for (const std::shared_ptr<Field> &field : input_fields_) {
      if (const InputField *input_field = dynamic_cast<const InputField *>(field.get())) {
        fn(*input_field);
      }
      else {
        field->foreach_input(fn);
      }
    }
  }

  void foreach_input(blender::FunctionRef<void(const InputField &input)> fn) final
  {
    for (const std::shared_ptr<Field> &field : input_fields_) {
      if (const InputField *input_field = dynamic_cast<const InputField *>(field.get())) {
        fn(*input_field);
      }
    }
  }
};

void add_procedure_inputs_recursive(const Field &field, MFProcedureBuilder &builder)
{
  field.foreach_input()
}

/**
 * Evaluate more than one prodecure at a time
 */
void evaluate_fields(const Span<std::shared_ptr<Field>> fields,
                     const MutableSpan<GMutableSpan> outputs,
                     const IndexMask mask)
{
  blender::Map<const InputField *, GVArrayPtr> computed_inputs;
  for (const std::shared_ptr<Field> &field : fields) {
    field->foreach_input_recursive([&](const InputField &input_field) {
      if (!computed_inputs.contains(&input_field)) {
        computed_inputs.add_new(&input_field, input_field.get_data(mask));
      }
    });
  }

  /* Build procedure. */
  MFProcedure procedure;
  MFProcedureBuilder builder{procedure};

  Map<const InputField *, MFVariable *> fields_to_variables;

  /* Add the unique inputs. */
  for (blender::Map<const InputField *, GVArrayPtr>::Item item : computed_inputs.items()) {
    fields_to_variables.add_new(
        item.key, &builder.add_parameter(MFParamType::ForSingleInput(item.value->type())));
  }

  /* Add the inputs recursively for the entire group of nodes. */
  // builder.add_return();
  // builder.add_output_parameter(*var4);

  BLI_assert(procedure.validate());

  /* Evaluate procedure. */
  MFProcedureExecutor executor{"Evaluate Field", procedure};
  MFParamsBuilder params{executor, mask.min_array_size()};
  MFContextBuilder context;

  /* Add the input data. */
  for (blender::Map<const InputField *, GVArrayPtr>::Item item : computed_inputs.items()) {
    params.add_readonly_single_input(*item.value);
  }

  /* Add the output arrays. */
  for (const int i : fields.index_range()) {
    params.add_uninitialized_single_output(outputs[i]);
  }

  executor.call(mask, params, context);

  // int input_index = 0;
  // for (const int param_index : fn_->param_indices()) {
  //   fn::MFParamType param_type = fn_->param_type(param_index);
  //   switch (param_type.category()) {
  //     case fn::MFParamType::SingleInput: {
  //       const Field &field = *input_fields_[input_index];
  //       FieldOutput &output = scope.add_value(field.evaluate(mask, inputs), __func__);
  //       params.add_readonly_single_input(output.varray_ref());
  //       input_index++;
  //       break;
  //     }
  //     case fn::MFParamType::SingleOutput: {
  //       const CPPType &type = param_type.data_type().single_type();
  //       void *buffer = MEM_mallocN_aligned(
  //           mask.min_array_size() * type.size(), type.alignment(), __func__);
  //       GMutableSpan span{type, buffer, mask.min_array_size()};
  //       outputs.append(span);
  //       params.add_uninitialized_single_output(span);
  //       if (param_index == output_param_index_) {
  //         output_span_index = outputs.size() - 1;
  //       }
  //       break;
  //     }
  //     case fn::MFParamType::SingleMutable:
  //     case fn::MFParamType::VectorInput:
  //     case fn::MFParamType::VectorMutable:
  //     case fn::MFParamType::VectorOutput:
  //       BLI_assert_unreachable();
  //       break;
  //   }
  // }
}

}  // namespace blender::fn