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

#include <cstring>

#include "BLI_listbase.h"

#include "DNA_screen_types.h"

#include "DEG_depsgraph_query.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "RNA_access.h"

#include "spreadsheet_intern.hh"

#include "spreadsheet_data_source_geometry.hh"
#include "spreadsheet_intern.hh"
#include "spreadsheet_layout.hh"
#include "spreadsheet_row_filter.hh"

namespace blender::ed::spreadsheet {

template<typename OperationFn>
static void apply_filter_operation(const ColumnValues &values,
                                   OperationFn check_fn,
                                   MutableSpan<bool> rows_included)
{
  for (const int i : rows_included.index_range()) {
    if (!rows_included[i]) {
      continue;
    }
    CellValue cell_value;
    values.get_value(i, cell_value);
    if (!check_fn(cell_value)) {
      rows_included[i] = false;
    }
  }
}

static void apply_row_filter(const SpreadsheetLayout &spreadsheet_layout,
                             const SpreadSheetRowFilter &row_filter,
                             MutableSpan<bool> rows_included)
{
  for (const ColumnLayout &column : spreadsheet_layout.columns) {
    const ColumnValues &values = *column.values;
    if (values.name() != row_filter.column_name) {
      continue;
    }

    switch (values.type()) {
      case ColumnValueType::Int32: {
        const int value = row_filter.value_int;
        switch (row_filter.operation) {
          case SPREADSHEET_ROW_FILTER_EQUAL: {
            apply_filter_operation(
                values,
                [value](const CellValue &cell_value) -> bool {
                  return *cell_value.value_int == value;
                },
                rows_included);
            break;
          }
          case SPREADSHEET_ROW_FILTER_GREATER: {
            apply_filter_operation(
                values,
                [value](const CellValue &cell_value) -> bool {
                  return *cell_value.value_int > value;
                },
                rows_included);
            break;
          }
          case SPREADSHEET_ROW_FILTER_LESS: {
            apply_filter_operation(
                values,
                [value](const CellValue &cell_value) -> bool {
                  return *cell_value.value_int < value;
                },
                rows_included);
            break;
          }
        }
        break;
      }
      case ColumnValueType::Float: {
        const float value = row_filter.value_float;
        switch (row_filter.operation) {
          case SPREADSHEET_ROW_FILTER_EQUAL: {
            apply_filter_operation(
                values,
                [value](const CellValue &cell_value) -> bool {
                  return *cell_value.value_float == value;
                },
                rows_included);
            break;
          }
          case SPREADSHEET_ROW_FILTER_GREATER: {
            apply_filter_operation(
                values,
                [value](const CellValue &cell_value) -> bool {
                  return *cell_value.value_float > value;
                },
                rows_included);
            break;
          }
          case SPREADSHEET_ROW_FILTER_LESS: {
            apply_filter_operation(
                values,
                [value](const CellValue &cell_value) -> bool {
                  return *cell_value.value_float < value;
                },
                rows_included);
            break;
          }
        }
        break;
      }
      case ColumnValueType::Bool: {
        const bool value = (row_filter.flag & SPREADSHEET_ROW_FILTER_BOOL_VALUE) != 0;
        apply_filter_operation(
            values,
            [value](const CellValue &cell_value) -> bool {
              return *cell_value.value_bool == value;
            },
            rows_included);
        break;
      }
      default:
        break;
    }

    /* Only one column should have this name. */
    break;
  }
}

static bool use_original_object_selection_filter(const SpaceSpreadsheet &sspreadsheet,
                                                 const DataSource &data_source)
{
  if (sspreadsheet.filter_flag & SPREADSHEET_FILTER_SELECTED_ONLY) {
    if (const GeometryDataSource *geometry_data_source = dynamic_cast<const GeometryDataSource *>(
            &data_source)) {
      Object *object_eval = geometry_data_source->object_eval();
      Object *object_orig = DEG_get_original_object(object_eval);
      if (object_orig->type == OB_MESH) {
        if (object_orig->mode == OB_MODE_EDIT) {
          return true;
        }
      }
    }
  }
  return false;
}

static void indices_vector_from_bools(Span<bool> selection, Vector<int64_t> &indices)
{
  for (const int i : selection.index_range()) {
    if (selection[i]) {
      indices.append(i);
    }
  }
}

Span<int64_t> spreadsheet_filter_rows(const SpaceSpreadsheet &sspreadsheet,
                                      const SpreadsheetLayout &spreadsheet_layout,
                                      const DataSource &data_source,
                                      ResourceScope &scope)
{
  const int tot_rows = data_source.tot_rows();

  if (!(sspreadsheet.filter_flag & SPREADSHEET_FILTER_ENABLE)) {
    return IndexRange(tot_rows).as_span();
  }

  const bool use_selection = use_original_object_selection_filter(sspreadsheet, data_source);

  if (BLI_listbase_is_empty(&sspreadsheet.row_filters) && !use_selection) {
    return IndexRange(tot_rows).as_span();
  }

  Array<bool> rows_included(tot_rows, true);

  LISTBASE_FOREACH (const SpreadSheetRowFilter *, row_filter, &sspreadsheet.row_filters) {
    apply_row_filter(spreadsheet_layout, *row_filter, rows_included);
  }

  if (use_selection) {
    const GeometryDataSource *geometry_data_source = dynamic_cast<const GeometryDataSource *>(
        &data_source);

    geometry_data_source->apply_selection_filter(rows_included);
  }

  Vector<int64_t> &indices = scope.construct<Vector<int64_t>>(__func__);
  indices_vector_from_bools(rows_included, indices);

  return indices;
}

}  // namespace blender::ed::spreadsheet