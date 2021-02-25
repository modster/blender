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

#include "spreadsheet_draw.hh"

#include "BLI_function_ref.hh"

namespace blender::ed::spreadsheet {

class TextHeaderDrawer final : public HeaderDrawer {
 private:
  std::string text_;

 public:
  TextHeaderDrawer(std::string text) : text_(std::move(text))
  {
  }

  void draw_header(const HeaderDrawParams &params) const final;
};

class ConstantTextCellDrawer final : public CellDrawer {
 private:
  std::string text_;

 public:
  ConstantTextCellDrawer(std::string text) : text_(std::move(text))
  {
  }

  void draw_cell(const CellDrawParams &params) const final;
};

class FloatCellDrawer : public CellDrawer {
 private:
  FunctionRef<float(int index)> get_value_;

 public:
  FloatCellDrawer(FunctionRef<float(int index)> get_value) : get_value_(get_value)
  {
  }

  void draw_cell(const CellDrawParams &params) const final;
};

class IntCellDrawer : public CellDrawer {
 private:
  FunctionRef<int(int index)> get_value_;

 public:
  IntCellDrawer(FunctionRef<int(int index)> get_value) : get_value_(get_value)
  {
  }

  void draw_cell(const CellDrawParams &params) const final;
};

class BoolCellDrawer : public CellDrawer {
 private:
  FunctionRef<bool(int index)> get_value_;

 public:
  BoolCellDrawer(FunctionRef<bool(int index)> get_value) : get_value_(get_value)
  {
  }

  void draw_cell(const CellDrawParams &params) const final;
};

}  // namespace blender::ed::spreadsheet
