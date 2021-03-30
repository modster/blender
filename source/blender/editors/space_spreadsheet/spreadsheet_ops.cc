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

#include "BLI_listbase.h"

#include "MEM_guardedalloc.h"

#include "BKE_context.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "ED_screen.h"

#include "WM_api.h"
#include "WM_types.h"

#include "spreadsheet_intern.hh"

static int row_filter_add_exec(bContext *C, wmOperator *UNUSED(op))
{
  SpaceSpreadsheet *spreadsheet_space = CTX_wm_space_spreadsheet(C);

  SpreadSheetRowFilter *row_filter = (SpreadSheetRowFilter *)MEM_callocN(
      sizeof(SpreadSheetRowFilter), __func__);

  row_filter->value_color[0] = 1.0f;
  row_filter->value_color[1] = 1.0f;
  row_filter->value_color[2] = 1.0f;
  row_filter->value_color[3] = 1.0f;
  row_filter->flag = (SPREADSHEET_ROW_FILTER_UI_EXPAND | SPREADSHEET_ROW_FILTER_ENABLED);

  BLI_addtail(&spreadsheet_space->row_filters, row_filter);

  ED_region_tag_redraw(CTX_wm_region(C));

  //   WM_event_add_notifier(C, NC_SPACE | ND..., );

  return OPERATOR_FINISHED;
}

static void SPREADSHEET_OT_add_rule(wmOperatorType *ot)
{
  ot->name = "Add Row Filter";
  ot->description = "Add a filter to remove rows from the displayed data";
  ot->idname = "SPREADSHEET_OT_add_rule";

  ot->exec = row_filter_add_exec;
  ot->poll = ED_operator_spreadsheet_active;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;
}

static int row_filter_remove_exec(bContext *C, wmOperator *op)
{
  SpaceSpreadsheet *spreadsheet_space = CTX_wm_space_spreadsheet(C);

  const int index = RNA_int_get(op->ptr, "index");

  SpreadSheetRowFilter *row_filter = (SpreadSheetRowFilter *)BLI_findlink(
      &spreadsheet_space->row_filters, index);
  if (row_filter == nullptr) {
    return OPERATOR_CANCELLED;
  }

  BLI_remlink(&spreadsheet_space->row_filters, row_filter);

  ED_region_tag_redraw(CTX_wm_region(C));

  //   WM_event_add_notifier(C, NC_SPACE | ND..., );

  return OPERATOR_FINISHED;
}

static void SPREADSHEET_OT_remove_rule(wmOperatorType *ot)
{
  ot->name = "Remove Row Filter";
  ot->description = "Remove a row filter from the rules";
  ot->idname = "SPREADSHEET_OT_remove_rule";

  ot->exec = row_filter_remove_exec;
  ot->poll = ED_operator_spreadsheet_active;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_int(ot->srna, "index", 0, 0, INT_MAX, "Index", "", 0, INT_MAX);
}

void spreadsheet_operatortypes()
{
  WM_operatortype_append(SPREADSHEET_OT_add_rule);
  WM_operatortype_append(SPREADSHEET_OT_remove_rule);
}
