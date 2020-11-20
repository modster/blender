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

/** \file
 * \ingroup edasset
 */

#include "BKE_asset.h"
#include "BKE_context.h"
#include "BKE_report.h"

#include "BLI_listbase.h"
#include "BLI_string_utils.h"

#include "DNA_asset_types.h"

#include "ED_asset.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "WM_api.h"
#include "WM_types.h"

static bool asset_make_poll(bContext *C)
{
  int tot_selected = 0;
  bool can_make_asset = false;

  /* Note that this isn't entirely cheap. Iterates over entire Outliner tree and allocates a link
   * for each selected item. The button only shows in the context menu though, so acceptable. */
  CTX_DATA_BEGIN (C, ID *, id, selected_ids) {
    tot_selected++;
    if (!id->asset_data) {
      can_make_asset = true;
      break;
    }
  }
  CTX_DATA_END;

  if (!can_make_asset) {
    if (tot_selected > 0) {
      CTX_wm_operator_poll_msg_set(C, "Selected data-blocks are already assets.");
    }
    else {
      CTX_wm_operator_poll_msg_set(C, "No data-blocks selected");
    }
    return false;
  }

  return true;
}

static int asset_make_exec(bContext *C, wmOperator *op)
{
  ID *last_id = NULL;
  int tot_created = 0;

  CTX_DATA_BEGIN (C, ID *, id, selected_ids) {
    if (id->asset_data) {
      continue;
    }

    ED_asset_make_for_id(C, id);
    last_id = id;
    tot_created++;
  }
  CTX_DATA_END;

  /* User feedback. */
  if (tot_created < 1) {
    BKE_report(op->reports, RPT_ERROR, "No data-blocks to create assets for found");
    return OPERATOR_CANCELLED;
  }
  if (tot_created == 1) {
    /* If only one data-block: Give more useful message by printing asset name. */
    BKE_reportf(op->reports, RPT_INFO, "Data-block '%s' is now an asset", last_id->name + 2);
  }
  else {
    BKE_reportf(op->reports, RPT_INFO, "%i data-blocks are now assets", tot_created);
  }

  WM_main_add_notifier(NC_ID | NA_EDITED, NULL);
  WM_main_add_notifier(NC_ASSET | NA_ADDED, NULL);

  return OPERATOR_FINISHED;
}

static void ASSET_OT_make(wmOperatorType *ot)
{
  ot->name = "Make Asset";
  ot->description = "Enable asset management for a data-block";
  ot->idname = "ASSET_OT_make";

  ot->poll = asset_make_poll;
  ot->exec = asset_make_exec;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;
}

static int asset_catalog_add_exec(bContext *C, wmOperator *op)
{
  AssetRepositoryInfo *repository_info = BKE_asset_repository_info_global_ensure();
  char name[MAX_NAME];

  RNA_string_get(op->ptr, "name", name);

  AssetCatalog *catalog = BKE_asset_repository_catalog_create(name);
  BLI_addtail(&repository_info->catalogs, catalog);
  BLI_uniquename(&repository_info->catalogs,
                 catalog,
                 name,
                 '.',
                 offsetof(AssetCatalog, name),
                 sizeof(catalog->name));

  /* Use WM notifier for now to denote a global data change. */
  WM_event_add_notifier(C, NC_WM | ND_DATACHANGED, NULL);

  return OPERATOR_FINISHED;
}

static void ASSET_OT_catalog_add(wmOperatorType *ot)
{
  ot->name = "Add Asset Catalog";
  ot->description =
      "Create a new asset catalog (a preset of filter settings to define which assets to display)";
  ot->idname = "ASSET_OT_catalog_add";

  /* TODO Popup should be removed once there's a proper UI to show and edit the catalog names. */
  ot->invoke = WM_operator_props_popup_confirm;
  ot->exec = asset_catalog_add_exec;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_string(
      ot->srna, "name", "Catalog", MAX_NAME, "Name", "Custom identifier for the catalog");
}

/* -------------------------------------------------------------------- */

void ED_operatortypes_asset(void)
{
  WM_operatortype_append(ASSET_OT_make);

  WM_operatortype_append(ASSET_OT_catalog_add);
}
