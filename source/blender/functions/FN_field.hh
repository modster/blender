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
 * For now, fields are very tied to the multi-function system, but in the future the #FieldFunction
 * class could be extended to use different descriptions of its outputs and computation besides
 * the embedded multi-function.
 */

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "FN_generic_virtual_array.hh"
#include "FN_multi_function_procedure.hh"
#include "FN_multi_function_procedure_builder.hh"
#include "FN_multi_function_procedure_executor.hh"

namespace blender::fn {

class FieldInput;
class FieldFunction;

/**
 * Descibes the output of a function. Generally corresponds to the combination of an output socket
 * and link combination in a node graph.
 */
class GField {
  /**
   * The function that calculates this field's values. Many fields can share the same function,
   * since a function can have many outputs, just like a node graph, where a single output can be
   * used as multiple inputs. This avoids calling the same function many times, only using one of
   * its results.
   */
  std::shared_ptr<FieldFunction> function_;
  /**
   * Which output of the function this field corresponds to.
   */
  int output_index_;

  std::shared_ptr<FieldInput> input_;

 public:
  GField(std::shared_ptr<FieldFunction> function, const int output_index)
      : function_(std::move(function)), output_index_(output_index)
  {
  }

  GField(std::shared_ptr<FieldInput> input) : input_(std::move(input))
  {
  }

  const fn::CPPType &cpp_type() const;

  bool is_input() const
  {
    return input_.get() != nullptr;
  }
  const FieldInput &input() const
  {
    BLI_assert(!function_);
    BLI_assert(input_);
    return *input_;
  }

  bool is_function() const
  {
    return function_.get() != nullptr;
  }
  const FieldFunction &function() const
  {
    BLI_assert(function_ != nullptr);
    BLI_assert(input_ == nullptr);
    return *function_;
  }

  int function_output_index() const
  {
    BLI_assert(function_ != nullptr);
    BLI_assert(input_ == nullptr);
    return output_index_;
  }
};

template<typename T> class Field {
 private:
  GField field_;

 public:
  Field(GField field) : field_(std::move(field))
  {
    BLI_assert(field_.cpp_type().is<T>());
  }

  const GField *operator->() const
  {
    return &field_;
  }

  const GField &operator*() const
  {
    return field_;
  }
};

/**
 * An operation acting on data described by fields. Generally corresponds
 * to a node or a subset of a node in a node graph.
 */
class FieldFunction {
  /**
   * The function used to calculate the field.
   */
  std::unique_ptr<const MultiFunction> owned_function_;
  const MultiFunction *function_;

  /**
   * References to descriptions of the results from the functions this function depends on.
   */
  blender::Vector<GField> inputs_;

 public:
  FieldFunction(std::unique_ptr<const MultiFunction> function, Vector<GField> &&inputs)
      : owned_function_(std::move(function)), inputs_(std::move(inputs))
  {
    function_ = owned_function_.get();
  }

  Span<GField> inputs() const
  {
    return inputs_;
  }

  const MultiFunction &multi_function() const
  {
    return *function_;
  }

  const CPPType &cpp_type_of_output_index(int index) const
  {
    MFParamType param_type = function_->param_type(index);
    MFDataType data_type = param_type.data_type();
    BLI_assert(param_type.interface_type() == MFParamType::Output);
    BLI_assert(data_type.is_single());
    return data_type.single_type();
  }
};

class FieldInput {
 protected:
  const CPPType *type_;
  std::string debug_name_;

 public:
  FieldInput(const CPPType &type, std::string debug_name = "")
      : type_(&type), debug_name_(std::move(debug_name))
  {
  }

  virtual GVArrayPtr get_varray_generic_context(IndexMask mask) const = 0;

  blender::StringRef debug_name() const
  {
    return debug_name_;
  }

  const CPPType &cpp_type() const
  {
    return *type_;
  }
};

/**
 * Evaluate more than one field at a time, as an optimization
 * in case they share inputs or various intermediate values.
 */
void evaluate_fields(blender::Span<GField> fields,
                     blender::IndexMask mask,
                     blender::Span<GMutableSpan> outputs);

/* --------------------------------------------------------------------
 * GField inline methods.
 */

inline const CPPType &GField::cpp_type() const
{
  if (this->is_function()) {
    return function_->cpp_type_of_output_index(output_index_);
  }
  return input_->cpp_type();
}

}  // namespace blender::fn
