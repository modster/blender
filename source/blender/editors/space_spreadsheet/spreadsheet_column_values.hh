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

#include "BLI_string_ref.hh"

#include "spreadsheet_cell_value.hh"

namespace blender::ed::spreadsheet {

enum class ColumnValueType {
  Int32,
  Float,
  Bool,
  Instances,
};

/**
 * This represents a column in a spreadsheet. It has a name and provides a value for all the cells
 * in the column.
 */
class ColumnValues {
 protected:
  ColumnValueType type_;
  std::string name_;
  int size_;

 public:
  ColumnValues(const ColumnValueType type, std::string name, const int size)
      : type_(type), name_(std::move(name)), size_(size)
  {
  }

  virtual ~ColumnValues() = default;

  virtual void get_value(int index, CellValue &r_cell_value) const = 0;

  ColumnValueType type() const
  {
    return type_;
  }

  StringRefNull name() const
  {
    return name_;
  }

  int size() const
  {
    return size_;
  }

  /* The default width of newly created columns, in UI units. */
  float default_width = 0.0f;
};

/* Utility class for the function below. */
template<typename GetValueF> class LambdaColumnValues : public ColumnValues {
 private:
  GetValueF get_value_;

 public:
  LambdaColumnValues(const ColumnValueType type, std::string name, int size, GetValueF get_value)
      : ColumnValues(type, std::move(name), size), get_value_(std::move(get_value))
  {
  }

  void get_value(int index, CellValue &r_cell_value) const final
  {
    get_value_(index, r_cell_value);
  }
};

/* Utility function that simplifies creating a spreadsheet column from a lambda function. */
template<typename GetValueF>
std::unique_ptr<ColumnValues> column_values_from_function(const ColumnValueType type,
                                                          std::string name,
                                                          int size,
                                                          GetValueF get_value)
{
  return std::make_unique<LambdaColumnValues<GetValueF>>(
      type, std::move(name), size, std::move(get_value));
}

}  // namespace blender::ed::spreadsheet
