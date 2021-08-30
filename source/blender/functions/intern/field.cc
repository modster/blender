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
#include "BLI_stack.hh"

#include "FN_field.hh"

struct InputOrFunction {
  const void *ptr;

 public:
  InputOrFunction(const blender::fn::FieldFunction &function) : ptr(&function)
  {
  }
  InputOrFunction(const blender::fn::FieldInput &input) : ptr(&input)
  {
  }
  InputOrFunction(const blender::fn::Field &field) /* Maybe this is too clever. */
  {
    if (field.is_function()) {
      ptr = &field.function();
    }
    else {
      ptr = &field.input();
    }
  }
  friend bool operator==(const InputOrFunction &a, const InputOrFunction &b)
  {
    return a.ptr == b.ptr;
  }
};

template<> struct blender::DefaultHash<InputOrFunction> {
  uint64_t operator()(const InputOrFunction &value) const
  {
    return DefaultHash<const void *>{}(value.ptr);
  }
};

namespace blender::fn {

/**
 * A map to hold the output variables for each function or input so they can be reused.
 */
// using VariableMap = Map<const FunctionOrInput *, Vector<MFVariable *>>;

/**
 * TODO: This exists because it seemed helpful for the procedure creation to be able to store
 * mutable data for each input or function output. That still may be helpful in the future, but
 * currently it isn't useful.
 */
struct FieldVariable {
  MFVariable *variable;
  FieldVariable(MFVariable &variable) : variable(&variable)
  {
  }
};

using VariableMap = Map<InputOrFunction, Vector<FieldVariable>>;

/**
 * A map of the computed inputs for all of a field system's inputs, to avoid creating duplicates.
 * Usually virtual arrays are just references, but sometimes they can be heavier as well.
 */
using ComputedInputMap = Map<const MFVariable *, GVArrayPtr>;

static FieldVariable &get_field_variable(const Field &field, VariableMap &unique_variables)
{
  if (field.is_input()) {
    const FieldInput &input = field.input();
    return unique_variables.lookup(input).first();
  }
  const FieldFunction &function = field.function();
  MutableSpan<FieldVariable> function_outputs = unique_variables.lookup(function);
  return function_outputs[field.function_output_index()];
}

static const FieldVariable &get_field_variable(const Field &field,
                                               const VariableMap &unique_variables)
{
  if (field.is_input()) {
    const FieldInput &input = field.input();
    return unique_variables.lookup(input).first();
  }
  const FieldFunction &function = field.function();
  Span<FieldVariable> function_outputs = unique_variables.lookup(function);
  return function_outputs[field.function_output_index()];
}

/**
 * TODO: Merge duplicate input nodes, not just fields pointing to the same FieldInput.
 */
static void add_variables_for_input(const Field &field,
                                    Stack<const Field *> &fields_to_visit,
                                    MFProcedureBuilder &builder,
                                    VariableMap &unique_variables)
{
  fields_to_visit.pop();
  const FieldInput &input = field.input();
  MFVariable &variable = builder.add_input_parameter(MFDataType::ForSingle(field.type()),
                                                     input.name());
  unique_variables.add(input, {variable});
}

static void add_variables_for_function(const Field &field,
                                       Stack<const Field *> &fields_to_visit,
                                       MFProcedureBuilder &builder,
                                       VariableMap &unique_variables)
{
  const FieldFunction &function = field.function();
  for (const Field &input_field : function.inputs()) {
    if (!unique_variables.contains(input_field)) {
      /* The field for this input hasn't been handled yet. Handle it now, so that we know all
       * of this field's function inputs already have variables. TODO: Verify that this is the
       * best way to do a depth first traversal. These extra lookups don't seem ideal. */
      fields_to_visit.push(&input_field);
      return;
    }
  }

  fields_to_visit.pop();

  Vector<MFVariable *> inputs;
  Set<FieldVariable *> unique_inputs;
  for (const Field &input_field : function.inputs()) {
    FieldVariable &input = get_field_variable(input_field, unique_variables);
    unique_inputs.add(&input);
    inputs.append(input.variable);
  }

  Vector<MFVariable *> outputs = builder.add_call(function.multi_function(), inputs);
  Vector<FieldVariable> &unique_outputs = unique_variables.lookup_or_add(function, {});
  for (MFVariable *output : outputs) {
    unique_outputs.append(*output);
  }
}

static void add_unique_variables(const Span<Field> fields,
                                 MFProcedureBuilder &builder,
                                 VariableMap &unique_variables)
{
  Stack<const Field *> fields_to_visit;
  for (const Field &field : fields) {
    fields_to_visit.push(&field);
  }

  while (!fields_to_visit.is_empty()) {
    const Field &field = *fields_to_visit.peek();
    if (unique_variables.contains(field)) {
      fields_to_visit.pop();
      continue;
    }

    if (field.is_input()) {
      add_variables_for_input(field, fields_to_visit, builder, unique_variables);
    }
    else {
      add_variables_for_function(field, fields_to_visit, builder, unique_variables);
    }
  }
}

/**
 * Add destruct calls to the procedure so that internal variables and inputs are destructed before
 * the procedure finishes. Currently this just adds all of the destructs at the end. That is not
 * optimal, but properly ordering destructs should be combined with reordering function calls to
 * use variables more optimally.
 */
static void add_destructs(const Span<Field> fields,
                          MFProcedureBuilder &builder,
                          VariableMap &unique_variables)
{
  Set<InputOrFunction> outputs;
  for (const Field &field : fields) {
    outputs.add(field);
  }
  for (const InputOrFunction &input_or_function : unique_variables.keys()) {
    /* Don't destruct variables used for the output of the field network. */
    if (outputs.contains(input_or_function)) {
      continue;
    }
    for (FieldVariable &variable : unique_variables.lookup(input_or_function)) {
      builder.add_destruct(*variable.variable);
    }
  }
}

static void build_procedure(const Span<Field> fields,
                            MFProcedure &procedure,
                            VariableMap &unique_variables)
{
  MFProcedureBuilder builder{procedure};

  add_unique_variables(fields, builder, unique_variables);

  add_destructs(fields, builder, unique_variables);

  builder.add_return();

  for (const Field &field : fields) {
    MFVariable &input = *get_field_variable(field, unique_variables).variable;
    builder.add_output_parameter(input);
  }

  std::cout << procedure.to_dot();

  BLI_assert(procedure.validate());
}

/**
 * TODO: Maybe this doesn't add inputs in the same order as the the unique
 * variable traversal. Add a test for that and fix it if it doesn't work.
 */
static void gather_inputs(const Span<Field> fields,
                          const VariableMap &unique_variables,
                          const IndexMask mask,
                          MFParamsBuilder &params,
                          Vector<GVArrayPtr> &r_inputs)
{
  Set<const MFVariable *> computed_inputs;
  Stack<const Field *> fields_to_visit;
  for (const Field &field : fields) {
    fields_to_visit.push(&field);
  }

  while (!fields_to_visit.is_empty()) {
    const Field &field = *fields_to_visit.pop();
    if (field.is_input()) {
      const FieldInput &input = field.input();
      const FieldVariable &variable = get_field_variable(field, unique_variables);
      if (!computed_inputs.contains(variable.variable)) {
        GVArrayPtr data = input.retrieve_data(mask);
        computed_inputs.add_new(variable.variable);
        params.add_readonly_single_input(*data, input.name());
        r_inputs.append(std::move(data));
      }
    }
    else {
      const FieldFunction &function = field.function();
      for (const Field &input_field : function.inputs()) {
        fields_to_visit.push(&input_field);
      }
    }
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
  VariableMap unique_variables;
  build_procedure(fields, procedure, unique_variables);

  MFProcedureExecutor executor{"Evaluate Field", procedure};
  MFParamsBuilder params{executor, mask.min_array_size()};
  MFContextBuilder context;

  Vector<GVArrayPtr> inputs;
  gather_inputs(fields, unique_variables, mask, params, inputs);

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

  /* Process fields that just connect to inputs separately, since otherwise we need a special
   * case to avoid sharing the same variable for an input and output parameters elsewhere.
   * TODO: It would be nice if there were a more elegant way to handle this, rather than a
   * separate step here, but I haven't thought of anything yet. */
  Vector<Field> non_input_fields{fields};
  Vector<GMutableSpan> non_input_outputs{outputs};
  for (int i = fields.size() - 1; i >= 0; i--) {
    if (non_input_fields[i].is_input()) {
      non_input_fields[i].input().retrieve_data(mask)->materialize(mask, outputs[i].data());

      non_input_fields.remove_and_reorder(i);
      non_input_outputs.remove_and_reorder(i);
    }
  }

  if (!non_input_fields.is_empty()) {
    evaluate_non_input_fields(non_input_fields, mask, non_input_outputs);
  }
}

}  // namespace blender::fn
