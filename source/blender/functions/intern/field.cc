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
                                 Map<const Field *, MFVariable *> &output_map)
{
  /* Recursively make sure all of the inputs have entries in the variable map. */
  field.foreach_input_recursive(
      [&](const Field &input_field) { add_field_parameters(input_field, builder, output_map); });

  /* Add the immediate inputs to this field. */
  Vector<MFVariable *> inputs;
  field.foreach_input([&](const Field &input_field) {
    MFVariable *input = output_map.lookup(&input_field);
    builder.add_input_parameter(input->data_type());
    inputs.append(input);
  });

  Vector<MFVariable *> outputs = builder.add_call(field.function(), inputs);

  builder.add_destruct(inputs);

  /* TODO: How to support multiple outputs?! */
  BLI_assert(outputs.size() == 1);
  output_map.add_new(&field, outputs.first());
}

static void build_procedure(const Span<Field> fields, MFProcedure &procedure)
{
  MFProcedureBuilder builder{procedure};

  Map<const Field *, MFVariable *> output_map;

  for (const Field &field : fields) {
    add_field_parameters(field, builder, output_map);
  }

  builder.add_return();

  for (const Field &field : fields) {
    builder.add_output_parameter(*output_map.lookup(&field));
  }

  std::cout << procedure.to_dot();

  BLI_assert(procedure.validate());
}

static void evaluate_procedure(MFProcedure &procedure,
                               const IndexMask mask,
                               const MutableSpan<GMutableSpan> outputs)
{
  MFProcedureExecutor executor{"Evaluate Field", procedure};
  MFParamsBuilder params{executor, mask.min_array_size()};
  MFContextBuilder context;

  /* Add the output arrays. */
  for (const int i : outputs.index_range()) {
    params.add_uninitialized_single_output(outputs[i]);
  }

  executor.call(mask, params, context);
}

/**
 * Evaluate more than one prodecure at a time
 */
void evaluate_fields(const Span<Field> fields,
                     const MutableSpan<GMutableSpan> outputs,
                     const IndexMask mask)
{
  MFProcedure procedure;
  build_procedure(fields, procedure);

  evaluate_procedure(procedure, mask, outputs);
}

}  // namespace blender::fn