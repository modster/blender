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

#include "FN_field.hh"

namespace blender::fn {

static void add_field_parameters(const Field &field,
                                 MFProcedureBuilder &builder,
                                 Map<const Field *, MFVariable *> &variable_map)
{
  if (const MultiFunctionField *mf_field = dynamic_cast<const MultiFunctionField *>(&field)) {
    /* Recursively make sure all of the inputs have entries in the parameter map. */
    mf_field->foreach_input_recursive([&](const Field &input_field) {
      add_field_parameters(input_field, builder, variable_map);
    });

    /* Gather the immediate inputs to this field. */
    Vector<MFVariable *> inputs;
    mf_field->foreach_input(
        [&](const Field &input_field) { inputs.append(variable_map.lookup(&input_field)); });

    for (const int i : inputs.index_range()) {
      builder.add_input_parameter(inputs[i]->data_type());
    }

    Vector<MFVariable *> outputs = builder.add_call(mf_field->function());

    // builder.add_destruct(inputs);

    /* TODO: How to support multiple outputs?! */
    BLI_assert(outputs.size() == 1);
    variable_map.add_new(mf_field, outputs.first());
  }
  else if (const InputField *input_field = dynamic_cast<const InputField *>(&field)) {
    variable_map.add_new(input_field,
                         &builder.add_input_parameter(MFDataType::ForSingle(field.type()),
                                                      std::string(field.debug_name())));
  }
}

static void build_procedure(const Span<FieldPtr> fields, MFProcedure &procedure)
{
  MFProcedureBuilder builder{procedure};

  Map<const Field *, MFVariable *> variable_map;

  for (const FieldPtr &field : fields) {
    if (dynamic_cast<const InputField *>(field.get())) {
      continue;
    }

    add_field_parameters(*field, builder, variable_map);
  }

  /* TODO: Move this to the proper place. */
  for (MFVariable *variable : variable_map.values()) {
    builder.add_destruct(*variable);
  }

  builder.add_return();

  BLI_assert(procedure.validate());
}

static void evaluate_procedure(MFProcedure &procedure,
                               const Span<FieldPtr> fields,
                               const IndexMask mask,
                               const MutableSpan<GMutableSpan> outputs)
{
  MFProcedureExecutor executor{"Evaluate Field", procedure};
  MFParamsBuilder params{executor, mask.min_array_size()};
  MFContextBuilder context;

  /* Add the input data from the input fields. */
  Map<const InputField *, GVArrayPtr> computed_inputs;
  for (const FieldPtr &field : fields) {
    if (const InputField *input_field = dynamic_cast<const InputField *>(field.get())) {
      if (!computed_inputs.contains(input_field)) {
        computed_inputs.add_new(input_field, input_field->get_data(mask));
      }
      continue;
    }
    field->foreach_input_recursive([&](const Field &field) {
      if (const InputField *input_field = dynamic_cast<const InputField *>(&field)) {
        /* TODO: Optimize, too many lookups. */
        if (!computed_inputs.contains(input_field)) {
          computed_inputs.add_new(input_field, input_field->get_data(mask));
        }
        params.add_readonly_single_input(*computed_inputs.lookup(input_field));
      }
    });
  }

  /* Add the output arrays. */
  for (const int i : outputs.index_range()) {
    if (const InputField *input_field = dynamic_cast<const InputField *>(fields[i].get())) {
      computed_inputs.lookup(input_field)->materialize(mask, outputs[i].data());
    }
    else {
      params.add_uninitialized_single_output(outputs[i]);
    }
  }

  executor.call(mask, params, context);
}

/**
 * Evaluate more than one prodecure at a time
 */
void evaluate_fields(const Span<FieldPtr> fields,
                     const MutableSpan<GMutableSpan> outputs,
                     const IndexMask mask)
{
  MFProcedure procedure;
  build_procedure(fields, procedure);

  evaluate_procedure(procedure, fields, mask, outputs);
}

}  // namespace blender::fn