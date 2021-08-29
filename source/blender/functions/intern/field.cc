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
#include "BLI_set.hh"
#include "BLI_vector_set.hh"

#include "FN_field.hh"

namespace blender::fn {

/**
 * A map to hold the output variables for each function or input so they can be reused.
 */
// using VariableMap = Map<const FunctionOrInput *, Vector<MFVariable *>>;
using VariableMap = Map<const void *, Vector<MFVariable *>>;
/* TODO: Use use counter in the vector to control when to add the desctruct call. */

/**
 * A map of the computed inputs for all of a field system's inputs, to avoid creating duplicates.
 * Usually virtual arrays are just references, but sometimes they can be heavier as well.
 */
using ComputedInputMap = Map<const MFVariable *, GVArrayPtr>;

static MFVariable &get_field_variable(const Field &field, const VariableMap &variable_map)
{
  if (field.is_input()) {
    const FieldInput &input = field.input();
    return *variable_map.lookup(&input).first();
  }
  const FieldFunction &function = field.function();
  const Span<MFVariable *> function_outputs = variable_map.lookup(&function);
  return *function_outputs[field.function_output_index()];
}

/**
 * Traverse the fields recursively. Eventually there will be a field whose function has no
 * inputs. Start adding multi-function variables there. Those outputs are then used as inputs
 * for the dependent functions, and the rest of the field tree is built up from there.
 */
static void add_field_variables_recursive(const Field &field,
                                          MFProcedureBuilder &builder,
                                          VariableMap &variable_map)
{
  if (field.is_input()) {
    const FieldInput &input = field.input();
    if (!variable_map.contains(&input)) {
      MFVariable &variable = builder.add_input_parameter(MFDataType::ForSingle(field.type()),
                                                         input.name());
      variable_map.add(&input, {&variable});
    }
  }
  else {
    const FieldFunction &function = field.function();
    for (const Field &input_field : function.inputs()) {
      add_field_variables_recursive(input_field, builder, variable_map); /* TODO: Use stack. */
    }

    /* Add the immediate inputs to this field, which were added earlier in the recursive call.  */
    Vector<MFVariable *> inputs;
    VectorSet<MFVariable *> unique_inputs;
    for (const Field &input_field : function.inputs()) {
      MFVariable &input = get_field_variable(input_field, variable_map);
      unique_inputs.add(&input);
      inputs.append(&input);
    }

    Vector<MFVariable *> outputs = builder.add_call(function.multi_function(), inputs);

    builder.add_destruct(unique_inputs); /* TODO: What if the same variable was used later on? */

    variable_map.add(&function, std::move(outputs));
  }
}

static void build_procedure(const Span<Field> fields,
                            MFProcedure &procedure,
                            VariableMap &variable_map)
{
  MFProcedureBuilder builder{procedure};

  for (const Field &field : fields) {
    add_field_variables_recursive(field, builder, variable_map);
  }

  builder.add_return();

  for (const Field &field : fields) {
    MFVariable &input = get_field_variable(field, variable_map);
    builder.add_output_parameter(input);
  }

  std::cout << procedure.to_dot();

  BLI_assert(procedure.validate());
}

/**
 * \TODO: In the future this could remove from the input map instead of building a second map.
 * Right now it's preferrable to keep this more understandable though.
 */
static void gather_inputs_recursive(const Field &field,
                                    const VariableMap &variable_map,
                                    const IndexMask mask,
                                    MFParamsBuilder &params,
                                    Set<const MFVariable *> &computed_inputs,
                                    Vector<GVArrayPtr> &r_inputs)
{
  if (field.is_input()) {
    const FieldInput &input = field.input();
    const MFVariable *variable = variable_map.lookup(&input).first();
    if (!computed_inputs.contains(variable)) {
      GVArrayPtr data = input.retrieve_data(mask);
      computed_inputs.add_new(variable);
      params.add_readonly_single_input(*data, input.name());
      r_inputs.append(std::move(data));
    }
  }
  else {
    const FieldFunction &function = field.function();
    for (const Field &input_field : function.inputs()) {
      gather_inputs_recursive(input_field, variable_map, mask, params, computed_inputs, r_inputs);
    }
  }
}

static void gather_inputs(const Span<Field> fields,
                          const VariableMap &variable_map,
                          const IndexMask mask,
                          MFParamsBuilder &params,
                          Vector<GVArrayPtr> &r_inputs)
{
  Set<const MFVariable *> computed_inputs;
  for (const Field &field : fields) {
    gather_inputs_recursive(field, variable_map, mask, params, computed_inputs, r_inputs);
  }
}

static void add_outputs(MFParamsBuilder &params, Span<GMutableSpan> outputs)
{
  for (const int i : outputs.index_range()) {
    params.add_uninitialized_single_output(outputs[i]);
  }
}

static void evaluate_non_input_fields(const Span<Field> fields,
                                      const IndexMask mask,
                                      const Span<GMutableSpan> outputs)
{
  MFProcedure procedure;
  VariableMap variable_map;
  build_procedure(fields, procedure, variable_map);

  MFProcedureExecutor executor{"Evaluate Field", procedure};
  MFParamsBuilder params{executor, mask.min_array_size()};
  MFContextBuilder context;

  Vector<GVArrayPtr> inputs;
  gather_inputs(fields, variable_map, mask, params, inputs);

  add_outputs(params, outputs);

  executor.call(mask, params, context);
}

/**
 * Evaluate more than one prodecure at a time, since often intermediate results will be shared
 * between multiple final results, and the procedure evaluator can optimize for this case.
 */
void evaluate_fields(const Span<Field> fields,
                     const IndexMask mask,
                     const Span<GMutableSpan> outputs)
{
  BLI_assert(fields.size() == outputs.size());

  /* Process fields that just connect to inputs separately, since otherwise we need
   * special case to avoid sharing the same variable for an input and output elsewhere. */
  Vector<Field> non_input_fields{fields};
  Vector<GMutableSpan> non_input_outputs{outputs};
  for (int i = fields.size() - 1; i >= 0; i--) {
    if (non_input_fields[i].is_input()) {
      non_input_fields[i].input().retrieve_data(mask)->materialize(mask, outputs[i].data());

      non_input_fields.remove_and_reorder(i);
      non_input_outputs.remove_and_reorder(i);
    }
  }

  evaluate_non_input_fields(non_input_fields, mask, non_input_outputs);
}

}  // namespace blender::fn
