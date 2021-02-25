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

#include "BLI_vector.hh"

struct uiBlock;
struct rcti;
struct bContext;
struct ARegion;

namespace blender::ed::spreadsheet {

struct HeaderDrawParams {
  uiBlock *block;
  int xmin, ymin;
  int width, height;
};

class HeaderDrawer {
 public:
  virtual ~HeaderDrawer() = default;
  virtual void draw_header(const HeaderDrawParams &params) const = 0;
};

struct CellDrawParams {
  uiBlock *block;
  int xmin, ymin;
  int width, height;
  int index;
};

class CellDrawer {
 public:
  virtual ~CellDrawer() = default;
  virtual void draw_cell(const CellDrawParams &params) const = 0;
};

struct SpreadsheetColumnLayout {
  int width;
  const HeaderDrawer *header_drawer = nullptr;
  const CellDrawer *cell_drawer = nullptr;
};

struct SpreadsheetLayout {
  int index_column_width;
  int header_row_height;
  int row_height;
  int row_index_digits;
  Span<int64_t> visible_rows;
  Vector<SpreadsheetColumnLayout> columns;
};

void draw_spreadsheet_in_region(const bContext *C,
                                ARegion *region,
                                const SpreadsheetLayout &spreadsheet_layout);

}  // namespace blender::ed::spreadsheet
