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
 * Field serve as an intermediate representation for a calculation of a group of functions. Having
 * an intermediate representation is helpful mainly to separate the execution system from the
 * system that describes the necessary computations. Fields can be executed in different contexts,
 * and optimization might mean executing the fields differently based on some factors like the
 * number of elements.
 *
 * For now, fields are very tied to the multi-function system, but in the future the #FieldFunction
 * class could be extended to use different descriptions of its outputs and computation besides
 * the embedded multi-function.
 */

#include "BLI_vector.hh"

#include "FN_multi_function_procedure.hh"
#include "FN_multi_function_procedure_builder.hh"
#include "FN_multi_function_procedure_executor.hh"

namespace blender::fn {

class Field;

/**
 * An operation acting on data described by fields. Generally corresponds
 * to a node or a subset of a node in a node graph.
 */
class FieldFunction {
  /**
   * The function used to calculate the
   */
  std::unique_ptr<MultiFunction> function_;

  /**
   * References to descriptions of the results from the functions this function depends on.
   */
  blender::Vector<Field *> inputs_;

  std::string name_;

 public:
  FieldFunction(std::unique_ptr<MultiFunction> function,
                Span<Field *> inputs,
                std::string &&name = "")
      : function_(std::move(function)), inputs_(inputs), name_(std::move(name))
  {
  }

  Span<Field *> inputs() const
  {
    return inputs_;
  }

  const MultiFunction &multi_function() const
  {
    return *function_;
  }

  blender::StringRef name() const
  {
    return name_;
  }
};

class FieldInput {
  std::string name_;

 public:
  FieldInput(std::string &&name = "") : name_(std::move(name))
  {
  }

  virtual GVArrayPtr retrieve_data(IndexMask mask) const = 0;

  blender::StringRef name() const
  {
    return name_;
  }
};

/**
 * Descibes the output of a function. Generally corresponds to the combination of an output socket
 * and link combination in a node graph.
 */
class Field {
  /**
   * The type of this field's result.
   */
  const fn::CPPType *type_;

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
  Field(const fn::CPPType &type, std::shared_ptr<FieldFunction> function, const int output_index)
      : type_(&type), function_(function), output_index_(output_index)
  {
  }

  Field(const fn::CPPType &type, std::shared_ptr<FieldInput> input) : type_(&type), input_(input)
  {
  }

  const fn::CPPType &type() const
  {
    BLI_assert(type_ != nullptr);
    return *type_;
  }

  bool is_input() const
  {
    return input_ != nullptr;
  }
  const FieldInput &input() const
  {
    BLI_assert(function_ == nullptr);
    BLI_assert(input_ != nullptr);
    return *input_;
  }

  bool is_function() const
  {
    return function_ != nullptr;
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

  blender::StringRef name() const
  {
    if (this->is_function()) {
      return function_->name();
    }
    return input_->name();
  }
};

/**
 * Evaluate more than one field at a time, as an optimization
 * in case they share inputs or various intermediate values.
 */
void evaluate_fields(blender::Span<Field> fields,
                     blender::IndexMask mask,
                     blender::Span<GMutableSpan> outputs);

}  // namespace blender::fn