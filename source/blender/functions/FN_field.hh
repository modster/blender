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
 *
 * Field serve as an intermediate representation for calculation of a group of functions. Having
 * an intermediate representation is helpful mainly to separate the execution system from the
 * system that describes the necessary computations. Fields can be executed in different contexts,
 * and optimization might mean executing the fields differently based on some factors like the
 * number of elements.
 *
 * For now, fields are very tied to the multi-function system, but in the future #FieldOperation
 * could be extended to use different descriptions of its outputs and computation besides the
 * embedded multi-function.
 */

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "FN_generic_virtual_array.hh"
#include "FN_multi_function_builder.hh"
#include "FN_multi_function_procedure.hh"
#include "FN_multi_function_procedure_builder.hh"
#include "FN_multi_function_procedure_executor.hh"

namespace blender::fn {

class FieldNode {
 public:
  ~FieldNode() = default;

  virtual const CPPType &cpp_type_of_output_index(int output_index) const = 0;

  virtual bool is_input_node() const
  {
    return false;
  }

  virtual bool is_operation_node() const
  {
    return false;
  }

  virtual uint64_t hash() const
  {
    return get_default_hash(this);
  }

  friend bool operator==(const FieldNode &a, const FieldNode &b)
  {
    return a.is_equal_to(b);
  }

  virtual bool is_equal_to(const FieldNode &other) const
  {
    return this == &other;
  }
};

/** Common base class for fields to avoid declaring the same methods for #GField and #GFieldRef. */
template<typename NodePtr> class GFieldBase {
 protected:
  NodePtr node_ = nullptr;
  int node_output_index_ = 0;

  GFieldBase(NodePtr node, const int node_output_index)
      : node_(std::move(node)), node_output_index_(node_output_index)
  {
  }

 public:
  GFieldBase() = default;

  operator bool() const
  {
    return node_ != nullptr;
  }

  friend bool operator==(const GFieldBase &a, const GFieldBase &b)
  {
    return &*a.node_ == &*b.node_ && a.node_output_index_ == b.node_output_index_;
  }

  uint64_t hash() const
  {
    return get_default_hash_2(node_, node_output_index_);
  }

  const fn::CPPType &cpp_type() const
  {
    return node_->cpp_type_of_output_index(node_output_index_);
  }

  bool has_input_node() const
  {
    return node_->is_input_node();
  }

  bool has_operation_node() const
  {
    return node_->is_operation_node();
  }

  const FieldNode &node() const
  {
    return *node_;
  }

  int node_output_index() const
  {
    return node_output_index_;
  }
};

/**
 * Describes the output of a function. Generally corresponds to the combination of an output socket
 * and link combination in a node graph.
 */
class GField : public GFieldBase<std::shared_ptr<FieldNode>> {
 public:
  GField() = default;

  GField(std::shared_ptr<FieldNode> node, const int node_output_index = 0)
      : GFieldBase<std::shared_ptr<FieldNode>>(std::move(node), node_output_index)
  {
  }
};

/**
 * Same as #GField but is cheaper to copy/move around, because it does not contain a
 * #std::shared_ptr.
 */
class GFieldRef : public GFieldBase<const FieldNode *> {
 public:
  GFieldRef() = default;

  GFieldRef(const GField &field)
      : GFieldBase<const FieldNode *>(&field.node(), field.node_output_index())
  {
  }

  GFieldRef(const FieldNode &node, const int node_output_index = 0)
      : GFieldBase<const FieldNode *>(&node, node_output_index)
  {
  }
};

template<typename T> class Field : public GField {
 public:
  Field() = default;

  Field(GField field) : GField(std::move(field))
  {
    BLI_assert(this->cpp_type().template is<T>());
  }

  Field(std::shared_ptr<FieldNode> node, const int node_output_index = 0)
      : Field(GField(std::move(node), node_output_index))
  {
  }
};

class FieldOperation : public FieldNode {
  std::unique_ptr<const MultiFunction> owned_function_;
  const MultiFunction *function_;

  blender::Vector<GField> inputs_;

 public:
  FieldOperation(std::unique_ptr<const MultiFunction> function, Vector<GField> inputs = {})
      : owned_function_(std::move(function)), inputs_(std::move(inputs))
  {
    function_ = owned_function_.get();
  }

  FieldOperation(const MultiFunction &function, Vector<GField> inputs = {})
      : function_(&function), inputs_(std::move(inputs))
  {
  }

  Span<GField> inputs() const
  {
    return inputs_;
  }

  const MultiFunction &multi_function() const
  {
    return *function_;
  }

  bool is_operation_node() const override
  {
    return true;
  }

  const CPPType &cpp_type_of_output_index(int output_index) const override
  {
    int output_counter = 0;
    for (const int param_index : function_->param_indices()) {
      MFParamType param_type = function_->param_type(param_index);
      if (param_type.is_output()) {
        if (output_counter == output_index) {
          return param_type.data_type().single_type();
        }
        output_counter++;
      }
    }
    BLI_assert_unreachable();
    return CPPType::get<float>();
  }
};

class FieldInput;

class FieldContext {
 public:
  ~FieldContext() = default;

  virtual const GVArray *try_get_varray_for_context(const FieldInput &field_input,
                                                    IndexMask mask,
                                                    ResourceScope &scope) const;
};

class FieldInput : public FieldNode {
 protected:
  const CPPType *type_;
  std::string debug_name_;

 public:
  FieldInput(const CPPType &type, std::string debug_name = "")
      : type_(&type), debug_name_(std::move(debug_name))
  {
  }

  virtual const GVArray *try_get_varray_for_context(const FieldContext &context,
                                                    IndexMask mask,
                                                    ResourceScope &scope) const = 0;

  blender::StringRef debug_name() const
  {
    return debug_name_;
  }

  const CPPType &cpp_type() const
  {
    return *type_;
  }

  const CPPType &cpp_type_of_output_index(int output_index) const override
  {
    BLI_assert(output_index == 0);
    UNUSED_VARS_NDEBUG(output_index);
    return *type_;
  }

  bool is_input_node() const override
  {
    return true;
  }
};

Vector<const GVArray *> evaluate_fields(ResourceScope &scope,
                                        Span<GFieldRef> fields_to_evaluate,
                                        IndexMask mask,
                                        const FieldContext &context,
                                        Span<GVMutableArray *> dst_hints = {});

void evaluate_constant_field(const GField &field, void *r_value);

void evaluate_fields_to_spans(Span<GFieldRef> fields_to_evaluate,
                              IndexMask mask,
                              const FieldContext &context,
                              Span<GMutableSpan> out_spans);

Vector<int64_t> indices_from_selection(const VArray<bool> &selection);

template<typename T> T evaluate_constant_field(const Field<T> &field)
{
  T value;
  value.~T();
  evaluate_constant_field(field, &value);
  return value;
}

template<typename T> Field<T> make_constant_field(T value)
{
  auto constant_fn = std::make_unique<fn::CustomMF_Constant<T>>(std::forward<T>(value));
  auto operation = std::make_shared<FieldOperation>(std::move(constant_fn));
  return Field<T>{GField{std::move(operation), 0}};
}

class FieldEvaluator : NonMovable, NonCopyable {
 private:
  struct OutputPointerInfo {
    void *dst = nullptr;
    /* When a destination virtual array is provided for an input, this is
     * unnecessary, otherwise this is used to construct the required virtual array. */
    void (*set)(void *dst, const GVArray &varray, ResourceScope &scope) = nullptr;
  };

  ResourceScope scope_;
  const FieldContext &context_;
  const IndexMask mask_;
  Vector<GField> fields_to_evaluate_;
  Vector<GVMutableArray *> dst_hints_;
  Vector<const GVArray *> evaluated_varrays_;
  Vector<OutputPointerInfo> output_pointer_infos_;
  bool is_evaluated_ = false;

 public:
  /** Takes #mask by pointer because the mask has to live longer than the evaluator. */
  FieldEvaluator(const FieldContext &context, const IndexMask *mask)
      : context_(context), mask_(*mask)
  {
  }
  FieldEvaluator(const FieldContext &context, const int64_t size) : context_(context), mask_(size)
  {
  }

  /**
   * \param field: Field to add to the evaluator.
   * \param dst: Mutable virtual array that the evaluated result for this field is be written into.
   */
  int add_with_destination(GField field, GVMutableArray &dst)
  {
    const int field_index = fields_to_evaluate_.append_and_get_index(std::move(field));
    dst_hints_.append(&dst);
    output_pointer_infos_.append({});
    return field_index;
  }

  /** Same as #add_with_destination but typed. */
  template<typename T> int add_with_destination(Field<T> field, VMutableArray<T> &dst)
  {
    GVMutableArray &generic_dst_hint = scope_.construct<GVMutableArray_For_VMutableArray<T>>(
        __func__, dst);
    return this->add_with_destination(GField(std::move(field)), generic_dst_hint);
  }

  /**
   * \param field: Field to add to the evaluator.
   * \param dst: Mutable span that the evaluated result for this field is be written into.
   * \note: When the output may only be used as a single value, the version of this function with
   * a virtual array result array should be used.
   */
  int add_with_destination(GField field, GMutableSpan dst)
  {
    const int field_index = fields_to_evaluate_.append_and_get_index(std::move(field));
    dst_hints_.append(&scope_.construct<GVMutableArray_For_GMutableSpan>(__func__, dst));
    output_pointer_infos_.append({});
    return field_index;
  }

  /**
   * \param field: Field to add to the evaluator.
   * \param dst: Mutable span that the evaluated result for this field is be written into.
   * \note: When the output may only be used as a single value, the version of this function with
   * a virtual array result array should be used.
   */
  template<typename T> int add_with_destination(Field<T> field, MutableSpan<T> dst)
  {
    const int field_index = fields_to_evaluate_.append_and_get_index(std::move(field));
    dst_hints_.append(&scope_.construct<GVMutableArray_For_MutableSpan<T>>(__func__, dst));
    output_pointer_infos_.append({});
    return field_index;
  }

  int add(GField field, const GVArray **varray_ptr)
  {
    const int field_index = fields_to_evaluate_.append_and_get_index(std::move(field));
    dst_hints_.append(nullptr);
    output_pointer_infos_.append(OutputPointerInfo{
        varray_ptr, [](void *dst, const GVArray &varray, ResourceScope &UNUSED(scope)) {
          *(const GVArray **)dst = &varray;
        }});
    return field_index;
  }

  /**
   * \param field: Field to add to the evaluator.
   * \param varray_ptr: Once #evaluate is called, the resulting virtual array will be will be
   *   assigned to the given position.
   * \return Index of the field in the evaluator which can be used in the #get_evaluated methods.
   */
  template<typename T> int add(Field<T> field, const VArray<T> **varray_ptr)
  {
    const int field_index = fields_to_evaluate_.append_and_get_index(std::move(field));
    dst_hints_.append(nullptr);
    output_pointer_infos_.append(OutputPointerInfo{
        varray_ptr, [](void *dst, const GVArray &varray, ResourceScope &scope) {
          *(const VArray<T> **)dst = &*scope.construct<GVArray_Typed<T>>(__func__, varray);
        }});
    return field_index;
  }

  /**
   * \return Index of the field in the evaluator which can be used in the #get_evaluated methods.
   */
  int add(GField field)
  {
    const int field_index = fields_to_evaluate_.append_and_get_index(std::move(field));
    dst_hints_.append(nullptr);
    output_pointer_infos_.append({});
    return field_index;
  }

  /**
   * Evaluate all fields on the evaluator. This can only be called once.
   */
  void evaluate()
  {
    BLI_assert_msg(!is_evaluated_, "Cannot evaluate fields twice.");
    Array<GFieldRef> fields(fields_to_evaluate_.size());
    for (const int i : fields_to_evaluate_.index_range()) {
      fields[i] = fields_to_evaluate_[i];
    }
    evaluated_varrays_ = evaluate_fields(scope_, fields, mask_, context_, dst_hints_);
    BLI_assert(fields_to_evaluate_.size() == evaluated_varrays_.size());
    for (const int i : fields_to_evaluate_.index_range()) {
      OutputPointerInfo &info = output_pointer_infos_[i];
      if (info.dst != nullptr) {
        info.set(info.dst, *evaluated_varrays_[i], scope_);
      }
    }
    is_evaluated_ = true;
  }

  const GVArray &get_evaluated(const int field_index) const
  {
    BLI_assert(is_evaluated_);
    return *evaluated_varrays_[field_index];
  }

  template<typename T> const VArray<T> &get_evaluated(const int field_index)
  {
    const GVArray &varray = this->get_evaluated(field_index);
    GVArray_Typed<T> &typed_varray = scope_.construct<GVArray_Typed<T>>(__func__, varray);
    return *typed_varray;
  }

  /**
   * Retrieve the output of an evaluated boolean field and convert it to a mask, which can be used
   * to avoid calculations for unnecessary elements later on. The evaluator will own the indices in
   * some cases, so it must live at least as long as the returned mask.
   */
  IndexMask get_evaluated_as_mask(const int field_index)
  {
    const GVArray &varray = this->get_evaluated(field_index);
    GVArray_Typed<bool> typed_varray{varray};

    if (typed_varray->is_single()) {
      if (typed_varray->get_internal_single()) {
        return IndexRange(typed_varray.size());
      }
      return IndexRange(0);
    }

    return scope_.add_value(indices_from_selection(*typed_varray), __func__).as_span();
  }
};

}  // namespace blender::fn
