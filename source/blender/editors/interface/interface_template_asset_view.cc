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
 * \ingroup edinterface
 */

#include "DNA_userdef_types.h"

#include "BLI_string.h"

#include "MEM_guardedalloc.h"

#include "UI_interface.h"

#include "WM_api.h"

#include "interface_intern.h"

/* TODO temporary includes for palettes. */
#include "BKE_global.h"
#include "BKE_main.h"
#include "BKE_screen.h"
#include "DNA_brush_types.h"
#include "RNA_access.h"

static void asset_view_draw_item(uiList *UNUSED(ui_list),
                                 bContext *UNUSED(C),
                                 uiLayout *layout,
                                 PointerRNA *UNUSED(dataptr),
                                 PointerRNA *itemptr,
                                 int UNUSED(icon),
                                 PointerRNA *UNUSED(active_dataptr),
                                 const char *UNUSED(active_propname),
                                 int index,
                                 int UNUSED(flt_flag))
{
  uiBlock *block = uiLayoutGetBlock(layout);
  uiButColor *color_but = (uiButColor *)uiDefButR(block,
                                                  UI_BTYPE_COLOR,
                                                  0,
                                                  "",
                                                  0,
                                                  0,
                                                  UI_UNIT_X,
                                                  UI_UNIT_Y,
                                                  itemptr,
                                                  "color",
                                                  -1,
                                                  0.0,
                                                  1.0,
                                                  0.0,
                                                  0.0,
                                                  "");
  color_but->is_pallete_color = true;
  color_but->palette_color_index = index;
}

static uiListType *UI_UL_asset_view(void)
{
  uiListType *list_type = (uiListType *)MEM_callocN(sizeof(*list_type), __func__);

  BLI_strncpy(list_type->idname, "UI_UL_asset_view", sizeof(list_type->idname));
  list_type->draw_item = asset_view_draw_item;

  return list_type;
}

void ED_uilisttypes_ui()
{
  WM_uilisttype_add(UI_UL_asset_view());
}

void uiTemplateAssetView(uiLayout *layout, bContext *C)
{
  Palette *palette = (Palette *)CTX_data_main(C)->palettes.first;

  PointerRNA id_ptr;
  RNA_id_pointer_create(&palette->id, &id_ptr);

  PointerRNA colors_poin;
  RNA_pointer_create(&palette->id, &RNA_PaletteColors, palette, &colors_poin);
  uiTemplateList(layout,
                 C,
                 "UI_UL_asset_view",
                 "asset_view",
                 &id_ptr,
                 "colors",
                 &colors_poin,
                 "active_index",
                 nullptr,
                 0,
                 0,
                 UILST_LAYOUT_FLEXIBLE_GRID,
                 0,
                 false,
                 false);
}
