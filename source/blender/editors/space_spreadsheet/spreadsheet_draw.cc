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

#include "UI_interface.h"
#include "UI_resources.h"
#include "UI_view2d.h"

#include "GPU_immediate.h"

#include "DNA_screen_types.h"

#include "BLI_rect.h"

#include "spreadsheet_draw.hh"

namespace blender::ed::spreadsheet {

static void draw_index_column_background(const uint pos,
                                         const ARegion *region,
                                         const SpreadsheetLayout &spreadsheet_layout)
{
  immUniformThemeColorShade(TH_BACK, 11);
  immRecti(pos,
           0,
           region->winy - spreadsheet_layout.header_row_height,
           spreadsheet_layout.index_column_width,
           0);
}

static void draw_alternating_row_overlay(const uint pos,
                                         const int scroll_offset_y,
                                         const ARegion *region,
                                         const SpreadsheetLayout &spreadsheet_layout)
{
  immUniformThemeColor(TH_ROW_ALTERNATE);
  GPU_blend(GPU_BLEND_ALPHA);
  const int row_pair_height = spreadsheet_layout.row_height * 2;
  const int row_top_y = region->winy - spreadsheet_layout.header_row_height -
                        scroll_offset_y % row_pair_height;
  for (const int i : IndexRange(region->winy / row_pair_height + 1)) {
    int x_left = 0;
    int x_right = region->winx;
    int y_top = row_top_y - i * row_pair_height - spreadsheet_layout.row_height;
    int y_bottom = y_top - spreadsheet_layout.row_height;
    y_top = std::min(y_top, region->winy - spreadsheet_layout.header_row_height);
    y_bottom = std::min(y_bottom, region->winy - spreadsheet_layout.header_row_height);
    immRecti(pos, x_left, y_top, x_right, y_bottom);
  }
  GPU_blend(GPU_BLEND_NONE);
}

static void draw_header_row_background(const uint pos,
                                       const ARegion *region,
                                       const SpreadsheetLayout &spreadsheet_layout)
{
  immUniformThemeColorShade(TH_BACK, 11);
  immRecti(
      pos, 0, region->winy, region->winx, region->winy - spreadsheet_layout.header_row_height);
}

static void draw_separator_lines(const uint pos,
                                 const int scroll_offset_x,
                                 const ARegion *region,
                                 const SpreadsheetLayout &spreadsheet_layout)
{
  immUniformThemeColorShade(TH_BACK, -11);

  immBeginAtMost(GPU_PRIM_LINES, spreadsheet_layout.columns.size() * 2 + 4);

  /* Index column line. */
  immVertex2i(pos, spreadsheet_layout.index_column_width, region->winy);
  immVertex2i(pos, spreadsheet_layout.index_column_width, 0);

  /* Header row line. */
  immVertex2i(pos, 0, region->winy - spreadsheet_layout.header_row_height);
  immVertex2i(pos, region->winx, region->winy - spreadsheet_layout.header_row_height);

  /* Column separator lines. */
  int line_x = spreadsheet_layout.index_column_width - scroll_offset_x;
  for (const int i : spreadsheet_layout.columns.index_range()) {
    const SpreadsheetColumnLayout &column = spreadsheet_layout.columns[i];
    line_x += column.width;
    if (line_x >= spreadsheet_layout.index_column_width) {
      immVertex2i(pos, line_x, region->winy);
      immVertex2i(pos, line_x, 0);
    }
  }
  immEnd();
}

static void get_visible_rows(const SpreadsheetLayout &spreadsheet_layout,
                             const ARegion *region,
                             const int scroll_offset_y,
                             int *r_first_row,
                             int *r_max_visible_rows)
{
  *r_first_row = -scroll_offset_y / spreadsheet_layout.row_height;
  *r_max_visible_rows = region->winy / spreadsheet_layout.row_height + 1;
}

static void draw_row_indices(const int scroll_offset_y,
                             const bContext *C,
                             ARegion *region,
                             const SpreadsheetLayout &spreadsheet_layout)
{
  GPU_scissor_test(true);
  GPU_scissor(0,
              0,
              spreadsheet_layout.index_column_width,
              region->winy - spreadsheet_layout.header_row_height);

  uiBlock *indices_block = UI_block_begin(C, region, __func__, UI_EMBOSS_NONE);
  int first_row, max_visible_rows;
  get_visible_rows(spreadsheet_layout, region, scroll_offset_y, &first_row, &max_visible_rows);
  for (const int i : IndexRange(first_row, max_visible_rows)) {
    if (i >= spreadsheet_layout.visible_rows.size()) {
      break;
    }
    const int index = spreadsheet_layout.visible_rows[i];
    const std::string index_str = std::to_string(index);
    const int x = 0;
    const int y = region->winy - spreadsheet_layout.header_row_height -
                  (i + 1) * spreadsheet_layout.row_height - scroll_offset_y;
    const int width = spreadsheet_layout.index_column_width;
    const int height = spreadsheet_layout.row_height;
    uiBut *but = uiDefIconTextBut(indices_block,
                                  UI_BTYPE_LABEL,
                                  0,
                                  ICON_NONE,
                                  index_str.c_str(),
                                  x,
                                  y,
                                  width,
                                  height,
                                  nullptr,
                                  0,
                                  0,
                                  0,
                                  0,
                                  nullptr);
    UI_but_drawflag_enable(but, UI_BUT_TEXT_RIGHT);
    UI_but_drawflag_disable(but, UI_BUT_TEXT_LEFT);
  }

  UI_block_end(C, indices_block);
  UI_block_draw(C, indices_block);

  GPU_scissor_test(false);
}

static void draw_column_headers(const bContext *C,
                                ARegion *region,
                                const SpreadsheetLayout &spreadsheet_layout,
                                const int scroll_offset_x)
{
  GPU_scissor_test(true);
  GPU_scissor(spreadsheet_layout.index_column_width + 1,
              region->winy - spreadsheet_layout.header_row_height,
              region->winx - spreadsheet_layout.index_column_width,
              spreadsheet_layout.header_row_height);

  uiBlock *column_headers_block = UI_block_begin(C, region, __func__, UI_EMBOSS_NONE);

  int left_x = spreadsheet_layout.index_column_width - scroll_offset_x;
  for (const int i : spreadsheet_layout.columns.index_range()) {
    const SpreadsheetColumnLayout &column_layout = spreadsheet_layout.columns[i];
    const int right_x = left_x + column_layout.width;

    if (column_layout.header_drawer != nullptr) {
      HeaderDrawParams params;
      params.block = column_headers_block;
      params.xmin = left_x;
      params.ymin = region->winy - spreadsheet_layout.header_row_height;
      params.width = column_layout.width;
      params.height = spreadsheet_layout.header_row_height;
      column_layout.header_drawer->draw_header(params);
    }

    left_x = right_x;
  }

  UI_block_end(C, column_headers_block);
  UI_block_draw(C, column_headers_block);

  GPU_scissor_test(false);
}

static void draw_cell_contents(const bContext *C,
                               ARegion *region,
                               const SpreadsheetLayout &spreadsheet_layout,
                               const int scroll_offset_x,
                               const int scroll_offset_y)
{
  GPU_scissor_test(true);
  GPU_scissor(spreadsheet_layout.index_column_width + 1,
              0,
              region->winx - spreadsheet_layout.index_column_width,
              region->winy - spreadsheet_layout.header_row_height);

  uiBlock *cells_block = UI_block_begin(C, region, __func__, UI_EMBOSS_NONE);

  int first_row, max_visible_rows;
  get_visible_rows(spreadsheet_layout, region, scroll_offset_y, &first_row, &max_visible_rows);

  int left_x = spreadsheet_layout.index_column_width - scroll_offset_x;
  for (const int column_index : spreadsheet_layout.columns.index_range()) {
    const SpreadsheetColumnLayout &column_layout = spreadsheet_layout.columns[column_index];
    const int right_x = left_x + column_layout.width;

    if (right_x >= spreadsheet_layout.index_column_width && left_x <= region->winx) {
      for (const int i : IndexRange(first_row, max_visible_rows)) {
        if (i >= spreadsheet_layout.visible_rows.size()) {
          break;
        }

        if (column_layout.cell_drawer != nullptr) {
          CellDrawParams params;
          params.block = cells_block;
          params.xmin = left_x;
          params.ymin = region->winy - spreadsheet_layout.header_row_height -
                        (i + 1) * spreadsheet_layout.row_height - scroll_offset_y;
          params.width = column_layout.width;
          params.height = spreadsheet_layout.row_height;
          params.index = spreadsheet_layout.visible_rows[i];
          column_layout.cell_drawer->draw_cell(params);
        }
      }
    }

    left_x = right_x;
  }

  UI_block_end(C, cells_block);
  UI_block_draw(C, cells_block);

  GPU_scissor_test(false);
}

static void update_view2d_tot_rect(const SpreadsheetLayout &spreadsheet_layout,
                                   ARegion *region,
                                   const int row_amount)
{
  int column_width_sum = 0;
  for (const SpreadsheetColumnLayout &column_layout : spreadsheet_layout.columns) {
    column_width_sum += column_layout.width;
  }
  UI_view2d_totRect_set(&region->v2d,
                        column_width_sum + spreadsheet_layout.index_column_width,
                        row_amount * spreadsheet_layout.row_height +
                            spreadsheet_layout.header_row_height);
}

void draw_spreadsheet_in_region(const bContext *C,
                                ARegion *region,
                                const SpreadsheetLayout &spreadsheet_layout)
{
  UI_ThemeClearColor(TH_BACK);

  View2D *v2d = &region->v2d;
  const int scroll_offset_y = v2d->cur.ymax;
  const int scroll_offset_x = v2d->cur.xmin;

  GPUVertFormat *format = immVertexFormat();
  uint pos = GPU_vertformat_attr_add(format, "pos", GPU_COMP_I32, 2, GPU_FETCH_INT_TO_FLOAT);
  immBindBuiltinProgram(GPU_SHADER_2D_UNIFORM_COLOR);

  draw_index_column_background(pos, region, spreadsheet_layout);
  draw_alternating_row_overlay(pos, scroll_offset_y, region, spreadsheet_layout);
  draw_header_row_background(pos, region, spreadsheet_layout);
  draw_separator_lines(pos, scroll_offset_x, region, spreadsheet_layout);

  immUnbindProgram();

  draw_row_indices(scroll_offset_y, C, region, spreadsheet_layout);
  draw_column_headers(C, region, spreadsheet_layout, scroll_offset_x);
  draw_cell_contents(C, region, spreadsheet_layout, scroll_offset_x, scroll_offset_y);

  update_view2d_tot_rect(spreadsheet_layout, region, spreadsheet_layout.visible_rows.size());

  rcti scroller_mask;
  BLI_rcti_init(&scroller_mask,
                spreadsheet_layout.index_column_width,
                region->winx,
                0,
                region->winy - spreadsheet_layout.header_row_height);
  UI_view2d_scrollers_draw(v2d, &scroller_mask);
}

}  // namespace blender::ed::spreadsheet
