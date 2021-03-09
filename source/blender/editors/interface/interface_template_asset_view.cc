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

#include "DNA_space_types.h"
#include "DNA_userdef_types.h"

#include "BLI_path_util.h"
#include "BLI_string.h"
#include "BLI_string_ref.hh"

#include "BLO_readfile.h"

#include "ED_asset.h"

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

struct AssetViewListData {
  AssetLibraryReference *asset_library;
  bScreen *screen;
};

static void asset_view_item_but_drag_set(uiBut *but,
                                         AssetViewListData *list_data,
                                         FileDirEntry *file)
{
  if (ID *id = file->id) {
    UI_but_drag_set_id(but, id);
  }
  else {
    const blender::StringRef asset_list_path = ED_assetlist_library_path(list_data->asset_library);
    char blend_path[FILE_MAX_LIBEXTRA];

    char path[FILE_MAX_LIBEXTRA];
    BLI_join_dirfile(path, sizeof(path), asset_list_path.data(), file->relpath);
    if (BLO_library_path_explode(path, blend_path, nullptr, nullptr)) {
      ImBuf *imbuf = ED_assetlist_asset_image_get(file);
      UI_but_drag_set_asset(but,
                            file->name,
                            BLI_strdup(blend_path),
                            file->blentype,
                            file->preview_icon_id,
                            imbuf,
                            1.0f);
    }
  }
}

static void asset_view_draw_item(uiList *ui_list,
                                 bContext *UNUSED(C),
                                 uiLayout *layout,
                                 PointerRNA *UNUSED(dataptr),
                                 PointerRNA *itemptr,
                                 int UNUSED(icon),
                                 PointerRNA *UNUSED(active_dataptr),
                                 const char *UNUSED(active_propname),
                                 int UNUSED(index),
                                 int UNUSED(flt_flag))
{
  AssetViewListData *list_data = (AssetViewListData *)ui_list->dyn_data->customdata;

  PropertyRNA *nameprop = RNA_struct_name_property(itemptr->type);
  char str[MAX_NAME];
  RNA_property_string_get(itemptr, nameprop, str);

  FileDirEntry *file = (FileDirEntry *)itemptr->data;
  uiBlock *block = uiLayoutGetBlock(layout);
  /* TODO ED_fileselect_init_layout(). Share somehow? */
  float size_x = (96.0f / 20.0f) * UI_UNIT_X;
  float size_y = (96.0f / 20.0f) * UI_UNIT_Y;
  uiBut *but = uiDefIconTextBut(block,
                                UI_BTYPE_PREVIEW_TILE,
                                0,
                                file->preview_icon_id,
                                file->name,
                                0,
                                0,
                                size_x,
                                size_y,
                                nullptr,
                                0,
                                0,
                                0,
                                0,
                                "");
  ui_def_but_icon(but, file->preview_icon_id, UI_HAS_ICON | UI_BUT_ICON_PREVIEW);
  asset_view_item_but_drag_set(but, list_data, file);
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

static void asset_view_template_list_item_iter_fn(PointerRNA *UNUSED(dataptr),
                                                  PropertyRNA *UNUSED(prop),
                                                  TemplateListIterData *iter_data,
                                                  uiTemplateListItemIterFn fn,
                                                  void *customdata)
{
  AssetViewListData *asset_iter_data = (AssetViewListData *)customdata;
  ED_assetlist_iterate(asset_iter_data->asset_library, [&](FileDirEntry &file) {
    PointerRNA itemptr;
    RNA_pointer_create(&asset_iter_data->screen->id, &RNA_FileSelectEntry, &file, &itemptr);
    fn(iter_data, &itemptr);
    return true;
  });
}

void uiTemplateAssetView(uiLayout *layout,
                         bContext *C,
                         PointerRNA *ptr,
                         const char *asset_library_propname,
                         const AssetFilterSettings *filter_settings)
{
  Palette *palette = (Palette *)CTX_data_main(C)->palettes.first;

  uiLayout *col = uiLayoutColumn(layout, false);

  PropertyRNA *asset_library_prop = RNA_struct_find_property(ptr, asset_library_propname);
  uiItemFullR(col, ptr, asset_library_prop, RNA_NO_INDEX, 0, 0, "", 0);

  AssetLibraryReference asset_library = ED_asset_library_reference_from_enum_value(
      RNA_property_enum_get(ptr, asset_library_prop));
  ED_assetlist_fetch(&asset_library, filter_settings, C);
  ED_assetlist_ensure_previews_job(&asset_library, C);

  AssetViewListData iter_data;
  iter_data.asset_library = &asset_library;
  iter_data.screen = CTX_wm_screen(C);

  /* TODO can we store more properties in the UIList? Asset specific filtering,  */
  /* TODO how can this be refreshed on updates? Maybe a notifier listener callback for the
   * uiListType? */
  PointerRNA colors_poin;
  RNA_pointer_create(&palette->id, &RNA_PaletteColors, palette, &colors_poin);
  PointerRNA rna_nullptr = PointerRNA_NULL;
  /* TODO can we have some kind of model-view API to handle referencing, filtering and lazy loading
   * (of previews) of the items? */
  uiTemplateList_ex(col,
                    C,
                    asset_view_template_list_item_iter_fn,
                    "UI_UL_asset_view",
                    "asset_view",
                    &rna_nullptr,
                    "",
                    &colors_poin,
                    "active_index",
                    nullptr,
                    0,
                    0,
                    UILST_LAYOUT_BIG_PREVIEW_GRID,
                    0,
                    false,
                    false,
                    &iter_data);
}
