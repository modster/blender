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

#include "BLI_map.hh"
#include "BLI_multi_value_map.hh"

#include "FN_field.hh"

namespace blender::fn {

/* A map to hold the output variables for each function so they can be reused. */
using OutputMap = MultiValueMap<const FieldFunction *, MFVariable *>;

static MFVariable *get_field_variable(const Field &field, const OutputMap &output_map)
{
  const FieldFunction &input_field_function = field.function();
  const Span<MFVariable *> input_function_outputs = output_map.lookup(&input_field_function);
  return input_function_outputs[field.function_output_index()];
}

/**
 * Traverse the fields recursively. Eventually there will be a field whose function has no
 * inputs. Start adding multi-function variables there. Those outputs are then used as inputs
 * for the dependent functions, and the rest of the field tree is built up from there.
 */
static void add_field_variables(const Field &field,
                                MFProcedureBuilder &builder,
                                OutputMap &output_map)
{
  const FieldFunction &function = field.function();
  for (const Field *input_field : function.inputs()) {
    add_field_variables(*input_field, builder, output_map);
  }

  /* Add the immediate inputs to this field, which were added before in the recursive call.
   * This will be skipped for functions with no inputs. */
  Vector<MFVariable *> inputs;
  for (const Field *input_field : function.inputs()) {
    MFVariable *input = get_field_variable(*input_field, output_map);
    builder.add_input_parameter(input->data_type());
    inputs.append(input);
  }

  Vector<MFVariable *> outputs = builder.add_call(function.multi_function(), inputs);

  builder.add_destruct(inputs);

  output_map.add_multiple(&function, outputs);
}

static void build_procedure(const Span<Field> fields, MFProcedure &procedure)
{
  MFProcedureBuilder builder{procedure};

  OutputMap output_map;

  for (const Field &field : fields) {
    add_field_variables(field, builder, output_map);
  }

  builder.add_return();

  for (const Field &field : fields) {
    MFVariable *input = get_field_variable(field, output_map);
    builder.add_output_parameter(*input);
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
                     const IndexMask mask,
                     const MutableSpan<GMutableSpan> outputs)
{
  MFProcedure procedure;
  build_procedure(fields, procedure);

  evaluate_procedure(procedure, mask, outputs);
}

}  // namespace blender::fn
