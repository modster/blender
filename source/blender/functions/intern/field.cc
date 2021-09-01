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
#include "BLI_set.hh"
#include "BLI_stack.hh"
#include "BLI_vector_set.hh"

#include "FN_field.hh"

namespace blender::fn {

struct GFieldRef {
  const FieldSource *source;
  int index;

  GFieldRef(const FieldSource &source, const int index) : source(&source), index(index)
  {
  }

  GFieldRef(const GField &field) : source(&field.source()), index(field.source_output_index())
  {
  }

  uint64_t hash() const
  {
    return get_default_hash_2(*source, index);
  }

  friend bool operator==(const GFieldRef &a, const GFieldRef &b)
  {
    return *a.source == *b.source && a.index == b.index;
  }
};

struct FieldGraphInfo {
  MultiValueMap<GFieldRef, const GField *> field_users;
  VectorSet<std::reference_wrapper<const ContextFieldSource>> deduplicated_context_sources;
};

static FieldGraphInfo preprocess_field_graph(Span<const GField *> entry_fields)
{
  FieldGraphInfo graph_info;

  Stack<const GField *> fields_to_check;
  Set<const GField *> handled_fields;

  for (const GField *field : entry_fields) {
    if (handled_fields.add(field)) {
      fields_to_check.push(field);
    }
  }

  while (!fields_to_check.is_empty()) {
    const GField *field = fields_to_check.pop();
    if (field->has_context_source()) {
      const ContextFieldSource &context_source = static_cast<const ContextFieldSource &>(
          field->source());
      graph_info.deduplicated_context_sources.add(context_source);
      continue;
    }
    BLI_assert(field->has_operation_source());
    const OperationFieldSource &operation = static_cast<const OperationFieldSource &>(
        field->source());
    for (const GField &operation_input : operation.inputs()) {
      graph_info.field_users.add(operation_input, field);
      if (handled_fields.add(&operation_input)) {
        fields_to_check.push(&operation_input);
      }
    }
  }
  return graph_info;
}

static Vector<const GVArray *> get_field_context_inputs(ResourceScope &scope,
                                                        const IndexMask mask,
                                                        FieldContext &context,
                                                        const FieldGraphInfo &graph_info)
{
  Vector<const GVArray *> field_context_inputs;
  for (const ContextFieldSource &context_source : graph_info.deduplicated_context_sources) {
    const GVArray *varray = context_source.try_get_varray_for_context(context, mask, scope);
    if (varray == nullptr) {
      const CPPType &type = context_source.cpp_type();
      varray = &scope.construct<GVArray_For_SingleValueRef>(
          __func__, type, mask.min_array_size(), type.default_value());
    }
    field_context_inputs.append(varray);
  }
  return field_context_inputs;
}

static Set<const GField *> find_varying_fields(const FieldGraphInfo &graph_info,
                                               Span<const GVArray *> field_context_inputs)
{
  Set<const GField *> found_fields;
  Stack<const GField *> fields_to_check;
  for (const int i : field_context_inputs.index_range()) {
    const GVArray *varray = field_context_inputs[i];
    if (varray->is_single()) {
      continue;
    }
    const ContextFieldSource &context_source = graph_info.deduplicated_context_sources[i];
    const GFieldRef context_source_field{context_source, 0};
    const Span<const GField *> users = graph_info.field_users.lookup(context_source_field);
    for (const GField *field : users) {
      if (found_fields.add(field)) {
        fields_to_check.push(field);
      }
    }
  }
  while (!fields_to_check.is_empty()) {
    const GField *field = fields_to_check.pop();
    const Span<const GField *> users = graph_info.field_users.lookup(*field);
    for (const GField *field : users) {
      if (found_fields.add(field)) {
        fields_to_check.push(field);
      }
    }
  }
  return found_fields;
}

static void build_multi_function_procedure_for_fields(MFProcedure &procedure,
                                                      const FieldGraphInfo &graph_info,
                                                      Span<const GField *> output_fields)
{
  MFProcedureBuilder builder{procedure};
  Map<GFieldRef, MFVariable *> variable_by_field;
  for (const ContextFieldSource &context_field_source : graph_info.deduplicated_context_sources) {
    MFVariable &variable = builder.add_input_parameter(
        MFDataType::ForSingle(context_field_source.cpp_type()), context_field_source.debug_name());
    variable_by_field.add_new({context_field_source, 0}, &variable);
  }

  struct FieldWithIndex {
    const GField *field;
    int current_input_index = 0;
  };

  for (const GField *field : output_fields) {
    Stack<FieldWithIndex> fields_to_check;
    fields_to_check.push({field, 0});
    while (!fields_to_check.is_empty()) {
      FieldWithIndex &field_with_index = fields_to_check.peek();
      const GField &field = *field_with_index.field;
      if (variable_by_field.contains(field)) {
        fields_to_check.pop();
        continue;
      }
      /* Context sources should already be handled above. */
      BLI_assert(field.has_operation_source());
      const OperationFieldSource &operation_field_source =
          static_cast<const OperationFieldSource &>(field.source());
      const Span<GField> operation_inputs = operation_field_source.inputs();
      if (field_with_index.current_input_index < operation_inputs.size()) {
        /* Push next input. */
        fields_to_check.push({&operation_inputs[field_with_index.current_input_index], 0});
        field_with_index.current_input_index++;
      }
      else {
        /* All inputs variables are ready, now add the function call. */
        Vector<MFVariable *> input_variables;
        for (const GField &field : operation_inputs) {
          input_variables.append(variable_by_field.lookup(field));
        }
        const MultiFunction &multi_function = operation_field_source.multi_function();
        Vector<MFVariable *> output_variables = builder.add_call(multi_function, input_variables);
        for (const int i : output_variables.index_range()) {
          variable_by_field.add_new({operation_field_source, i}, output_variables[i]);
        }
      }
    }
  }

  /* TODO: Handle case when there are duplicates in #output_fields. */
  for (const GField *field : output_fields) {
    MFVariable *variable = variable_by_field.lookup(*field);
    builder.add_output_parameter(*variable);
  }

  /* Remove the variables that should not be destructed from the map. */
  for (const GField *field : output_fields) {
    variable_by_field.remove(*field);
  }
  for (MFVariable *variable : variable_by_field.values()) {
    builder.add_destruct(*variable);
  }

  builder.add_return();

  // std::cout << procedure.to_dot() << "\n";
  BLI_assert(procedure.validate());
}

struct PartiallyInitializedArray : NonCopyable, NonMovable {
  void *buffer;
  IndexMask mask;
  const CPPType *type;

  ~PartiallyInitializedArray()
  {
    this->type->destruct_indices(this->buffer, this->mask);
  }
};

/**
 * Evaluate fields in the given context. If possible, multiple fields should be evaluated together,
 * because that can be more efficient when they share common sub-fields.
 *
 * \param scope: The resource scope that owns data that makes up the output virtual arrays. Make
 *   sure the scope is not destructed when the output virtual arrays are still used.
 * \param fields_to_evaluate: The fields that should be evaluated together.
 * \param mask: Determines which indices are computed. The mask may be referenced by the returned
 *   virtual arrays. So the underlying index span should live longer then #scope.
 * \param context: The context that the field is evaluated in.
 * \param dst_hints: If provided, the computed data will be written into those virtual arrays
 *   instead of into newly created ones. That allows making the computing data live longer
 *   than #scope and is more efficient when the data will be written into those virtual arrays
 *   later anyway.
 * \return The computed virtual arrays for each provided field. If #dst_hints were passed, the
 *   provided virtual arrays are returned.
 */
Vector<const GVArray *> evaluate_fields(ResourceScope &scope,
                                        Span<const GField *> fields_to_evaluate,
                                        IndexMask mask,
                                        FieldContext &context,
                                        Span<GVMutableArray *> dst_hints)
{
  Vector<const GVArray *> r_varrays(fields_to_evaluate.size(), nullptr);

  auto get_dst_hint_if_available = [&](int index) -> GVMutableArray * {
    if (dst_hints.is_empty()) {
      return nullptr;
    }
    return dst_hints[index];
  };

  FieldGraphInfo graph_info = preprocess_field_graph(fields_to_evaluate);
  Vector<const GVArray *> field_context_inputs = get_field_context_inputs(
      scope, mask, context, graph_info);

  /* Finish fields that output a context varray directly. */
  for (const int out_index : fields_to_evaluate.index_range()) {
    const GField &field = *fields_to_evaluate[out_index];
    if (!field.has_context_source()) {
      continue;
    }
    const ContextFieldSource &field_source = static_cast<const ContextFieldSource &>(
        field.source());
    const int field_source_index = graph_info.deduplicated_context_sources.index_of(field_source);
    const GVArray *varray = field_context_inputs[field_source_index];
    r_varrays[out_index] = varray;
  }

  Set<const GField *> varying_fields = find_varying_fields(graph_info, field_context_inputs);

  Vector<const GField *> varying_fields_to_evaluate;
  Vector<int> varying_field_indices;
  Vector<const GField *> constant_fields_to_evaluate;
  Vector<int> constant_field_indices;
  for (const int i : fields_to_evaluate.index_range()) {
    if (r_varrays[i] != nullptr) {
      /* Already done. */
      continue;
    }
    const GField *field = fields_to_evaluate[i];
    if (varying_fields.contains(field)) {
      varying_fields_to_evaluate.append(field);
      varying_field_indices.append(i);
    }
    else {
      constant_fields_to_evaluate.append(field);
      constant_field_indices.append(i);
    }
  }

  const int array_size = mask.min_array_size();
  if (!varying_fields_to_evaluate.is_empty()) {
    MFProcedure procedure;
    build_multi_function_procedure_for_fields(procedure, graph_info, varying_fields_to_evaluate);
    MFProcedureExecutor procedure_executor{"Procedure", procedure};
    MFParamsBuilder mf_params{procedure_executor, mask.min_array_size()};
    MFContextBuilder mf_context;

    for (const GVArray *varray : field_context_inputs) {
      mf_params.add_readonly_single_input(*varray);
    }

    for (const int i : varying_fields_to_evaluate.index_range()) {
      const GField *field = varying_fields_to_evaluate[i];
      const CPPType &type = field->cpp_type();
      const int out_index = varying_field_indices[i];

      GVMutableArray *output_varray = get_dst_hint_if_available(out_index);
      void *buffer;
      if (output_varray == nullptr || !output_varray->is_span()) {
        buffer = scope.linear_allocator().allocate(type.size() * array_size, type.alignment());

        PartiallyInitializedArray &destruct_helper = scope.construct<PartiallyInitializedArray>(
            __func__);
        destruct_helper.buffer = buffer;
        destruct_helper.mask = mask;
        destruct_helper.type = &type;

        r_varrays[out_index] = &scope.construct<GVArray_For_GSpan>(
            __func__, GSpan{type, buffer, array_size});
      }
      else {
        buffer = output_varray->get_internal_span().data();

        r_varrays[out_index] = output_varray;
      }

      const GMutableSpan span{type, buffer, array_size};
      mf_params.add_uninitialized_single_output(span);
    }

    procedure_executor.call(mask, mf_params, mf_context);
  }
  if (!constant_fields_to_evaluate.is_empty()) {
    MFProcedure procedure;
    build_multi_function_procedure_for_fields(procedure, graph_info, constant_fields_to_evaluate);
    MFProcedureExecutor procedure_executor{"Procedure", procedure};
    MFParamsBuilder mf_params{procedure_executor, 1};
    MFContextBuilder mf_context;

    for (const GVArray *varray : field_context_inputs) {
      mf_params.add_readonly_single_input(*varray);
    }

    Vector<GMutablePointer> output_buffers;
    for (const int i : constant_fields_to_evaluate.index_range()) {
      const GField *field = constant_fields_to_evaluate[i];
      const CPPType &type = field->cpp_type();
      void *buffer = scope.linear_allocator().allocate(type.size(), type.alignment());

      PartiallyInitializedArray &destruct_helper = scope.construct<PartiallyInitializedArray>(
          __func__);
      destruct_helper.buffer = buffer;
      destruct_helper.mask = IndexRange(1);
      destruct_helper.type = &type;

      mf_params.add_uninitialized_single_output({type, buffer, 1});

      const int out_index = constant_field_indices[i];
      r_varrays[out_index] = &scope.construct<GVArray_For_SingleValueRef>(
          __func__, type, array_size, buffer);
    }

    procedure_executor.call(IndexRange(1), mf_params, mf_context);
  }

  if (!dst_hints.is_empty()) {
    for (const int out_index : fields_to_evaluate.index_range()) {
      GVMutableArray *output_varray = get_dst_hint_if_available(out_index);
      if (output_varray == nullptr) {
        /* Caller did not provide a destination for this output. */
        continue;
      }
      const GVArray *computed_varray = r_varrays[out_index];
      BLI_assert(computed_varray->type() == output_varray->type());
      if (output_varray == computed_varray) {
        /* The result has been written into the destination provided by the caller already. */
        continue;
      }
      /* Still have to copy over the data in the destination provided by the caller. */
      if (output_varray->is_span()) {
        /* Materialize into a span. */
        computed_varray->materialize_to_uninitialized(output_varray->get_internal_span().data());
      }
      else {
        /* Slower materialize into a different structure. */
        const CPPType &type = computed_varray->type();
        BUFFER_FOR_CPP_TYPE_VALUE(type, buffer);
        for (const int i : mask) {
          computed_varray->get_to_uninitialized(i, buffer);
          output_varray->set_by_relocate(i, buffer);
        }
      }
      r_varrays[out_index] = output_varray;
    }
  }
  return r_varrays;
}

void evaluate_constant_field(const GField &field, void *r_value)
{
  ResourceScope scope;
  FieldContext context;
  Vector<const GVArray *> varrays = evaluate_fields(scope, {&field}, IndexRange(1), context);
  varrays[0]->get_to_uninitialized(0, r_value);
}

void evaluate_fields_to_spans(Span<const GField *> fields_to_evaluate,
                              IndexMask mask,
                              FieldContext &context,
                              Span<GMutableSpan> out_spans)
{
  ResourceScope scope;
  Vector<GVMutableArray *> varrays;
  for (GMutableSpan span : out_spans) {
    varrays.append(&scope.construct<GVMutableArray_For_GMutableSpan>(__func__, span));
  }
  evaluate_fields(scope, fields_to_evaluate, mask, context, varrays);
}

}  // namespace blender::fn
