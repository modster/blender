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

#include "FN_generic_virtual_array.hh"
#include "FN_multi_function_procedure.hh"
#include "FN_multi_function_procedure_builder.hh"
#include "FN_multi_function_procedure_executor.hh"

namespace blender::fn {

class Field;
using FieldPtr = std::unique_ptr<Field>;

class Field {
  const fn::CPPType *type_;
  std::string debug_name_ = "";

 public:
  virtual ~Field() = default;
  Field(const fn::CPPType &type, std::string &&debug_name = "")
      : type_(&type), debug_name_(std::move(debug_name))
  {
  }

  const fn::CPPType &type() const
  {
    BLI_assert(type_ != nullptr);
    return *type_;
  }

  blender::StringRef debug_name() const
  {
    return debug_name_;
  }

  virtual void foreach_input(blender::FunctionRef<void(const Field &input)> UNUSED(fn)) const = 0;
  virtual void foreach_input_recursive(
      blender::FunctionRef<void(const Field &input)> UNUSED(fn)) const = 0;
};

/**
 * A field that doesn't have any dependencies on other fields.
 *
 * TODO: It might be an elegant simplification if every single field was a multi-function field,
 * and input fields just happened to have no inputs. Then it might not need to be a virtual class,
 * since the dynamic behavior would be contained in the multifunction, which would be very nice.
 */
class InputField : public Field {
 public:
  InputField(const CPPType &type) : Field(type)
  {
  }

  void foreach_input(blender::FunctionRef<void(const Field &input)> UNUSED(fn)) const final
  {
  }
  void foreach_input_recursive(
      blender::FunctionRef<void(const Field &input)> UNUSED(fn)) const final
  {
  }

  virtual GVArrayPtr get_data(IndexMask mask) const = 0;

  /**
   * Return true when the field input is the same as another field, used as an
   * optimization to avoid creating multiple virtual arrays for the same input node.
   */
  virtual bool equals(const InputField &UNUSED(other))
  {
    return false;
  }
};

/**
 * A field that takes inputs
 */
class MultiFunctionField final : public Field {
  blender::Vector<FieldPtr> input_fields_;
  const MultiFunction *function_;

 public:
  void foreach_input(blender::FunctionRef<void(const Field &input)> fn) const final
  {
    for (const FieldPtr &field : input_fields_) {
      fn(*field);
    }
  }
  void foreach_input_recursive(blender::FunctionRef<void(const Field &input)> fn) const final
  {
    for (const FieldPtr &field : input_fields_) {
      fn(*field);
      field->foreach_input(fn);
    }
  }

  const MultiFunction &function() const
  {
    BLI_assert(function_ != nullptr);
    return *function_;
  }
};

/**
 * Evaluate more than one field at a time, as an optimization
 * in case they share inputs or various intermediate values.
 */
void evaluate_fields(const blender::Span<FieldPtr> fields,
                     const blender::MutableSpan<GMutableSpan> outputs,
                     const blender::IndexMask mask);

}  // namespace blender::fn