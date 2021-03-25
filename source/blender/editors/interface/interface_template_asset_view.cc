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

#include "BKE_screen.h"

#include "BLI_path_util.h"
#include "BLI_string.h"
#include "BLI_string_ref.hh"

#include "BLO_readfile.h"

#include "ED_asset.h"
#include "ED_screen.h"

#include "MEM_guardedalloc.h"

#include "RNA_access.h"

#include "UI_interface.h"

#include "WM_api.h"

#include "interface_intern.h"

struct AssetViewListData {
  AssetLibraryReference asset_library;
  bScreen *screen;
};

static void asset_view_item_but_drag_set(uiBut *but,
                                         AssetViewListData *list_data,
                                         AssetHandle *asset_handle)
{
  if (ID *id = asset_handle->file_data->id) {
    UI_but_drag_set_id(but, id);
  }
  else {
    const blender::StringRef asset_list_path = ED_assetlist_library_path(
        &list_data->asset_library);
    char blend_path[FILE_MAX_LIBEXTRA];

    char path[FILE_MAX_LIBEXTRA];
    BLI_join_dirfile(path, sizeof(path), asset_list_path.data(), asset_handle->file_data->relpath);
    if (BLO_library_path_explode(path, blend_path, nullptr, nullptr)) {
      ImBuf *imbuf = ED_assetlist_asset_image_get(asset_handle);
      UI_but_drag_set_asset(but,
                            asset_handle->file_data->name,
                            BLI_strdup(blend_path),
                            asset_handle->file_data->blentype,
                            asset_handle->file_data->preview_icon_id,
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

  BLI_assert(RNA_struct_is_a(itemptr->type, &RNA_AssetHandle));
  AssetHandle *asset_handle = (AssetHandle *)itemptr->data;

  uiLayoutSetContextPointer(layout, "asset_handle", itemptr);

  uiBlock *block = uiLayoutGetBlock(layout);
  /* TODO ED_fileselect_init_layout(). Share somehow? */
  float size_x = (96.0f / 20.0f) * UI_UNIT_X;
  float size_y = (96.0f / 20.0f) * UI_UNIT_Y;
  uiBut *but = uiDefIconTextBut(block,
                                UI_BTYPE_PREVIEW_TILE,
                                0,
                                asset_handle->file_data->preview_icon_id,
                                asset_handle->file_data->name,
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
  ui_def_but_icon(but,
                  asset_handle->file_data->preview_icon_id,
                  /* NOLINTNEXTLINE: bugprone-suspicious-enum-usage */
                  UI_HAS_ICON | UI_BUT_ICON_PREVIEW);
  if (!ui_list->custom_drag_opname) {
    asset_view_item_but_drag_set(but, list_data, asset_handle);
  }
}

static void asset_view_listener(uiList *ui_list, wmRegionListenerParams *params)
{
  AssetViewListData *list_data = (AssetViewListData *)ui_list->dyn_data->customdata;

  if (ED_assetlist_listen(&list_data->asset_library, params->notifier)) {
    ED_region_tag_redraw(params->region);
  }
}

static uiListType *UI_UL_asset_view()
{
  uiListType *list_type = (uiListType *)MEM_callocN(sizeof(*list_type), __func__);

  BLI_strncpy(list_type->idname, "UI_UL_asset_view", sizeof(list_type->idname));
  list_type->draw_item = asset_view_draw_item;
  list_type->listener = asset_view_listener;

  return list_type;
}

void ED_uilisttypes_ui()
{
  WM_uilisttype_add(UI_UL_asset_view());
}

static void asset_view_template_refresh_asset_collection(
    const AssetLibraryReference &asset_library,
    PointerRNA &assets_dataptr,
    const char *assets_propname)
{
  PropertyRNA *assets_prop = RNA_struct_find_property(&assets_dataptr, assets_propname);
  if (!assets_prop) {
    RNA_warning("Asset collection not found");
    return;
  }
  if (!RNA_struct_is_a(RNA_property_pointer_type(&assets_dataptr, assets_prop),
                       &RNA_AssetHandle)) {
    RNA_warning("Expected a collection property for AssetHandle items");
    return;
  }

  RNA_property_collection_clear(&assets_dataptr, assets_prop);

  ED_assetlist_iterate(&asset_library, [&](FileDirEntry &file) {
    PointerRNA itemptr, fileptr;
    RNA_property_collection_add(&assets_dataptr, assets_prop, &itemptr);

    RNA_pointer_create(nullptr, &RNA_FileSelectEntry, &file, &fileptr);
    RNA_pointer_set(&itemptr, "file_data", fileptr);

    /* Copy name from file to asset-handle name ID-property. */
    char name[MAX_NAME];
    PropertyRNA *file_name_prop = RNA_struct_name_property(fileptr.type);
    RNA_property_string_get(&fileptr, file_name_prop, name);
    PropertyRNA *asset_name_prop = RNA_struct_name_property(&RNA_AssetHandle);
    RNA_property_string_set(&itemptr, asset_name_prop, name);

    return true;
  });
}

void uiTemplateAssetView(uiLayout *layout,
                         bContext *C,
                         const char *list_id,
                         PointerRNA *asset_library_dataptr,
                         const char *asset_library_propname,
                         PointerRNA *assets_dataptr,
                         const char *assets_propname,
                         PointerRNA *active_dataptr,
                         const char *active_propname,
                         const AssetFilterSettings *filter_settings,
                         const char *activate_opname,
                         const char *drag_opname)
{
  if (!list_id || !list_id[0]) {
    RNA_warning("Asset view needs a valid identifier");
    return;
  }

  uiLayout *col = uiLayoutColumn(layout, false);

  PropertyRNA *asset_library_prop = RNA_struct_find_property(asset_library_dataptr,
                                                             asset_library_propname);
  uiItemFullR(col, asset_library_dataptr, asset_library_prop, RNA_NO_INDEX, 0, 0, "", 0);

  AssetLibraryReference asset_library = ED_asset_library_reference_from_enum_value(
      RNA_property_enum_get(asset_library_dataptr, asset_library_prop));
  ED_assetlist_storage_fetch(&asset_library, filter_settings, C);
  ED_assetlist_ensure_previews_job(&asset_library, C);

  asset_view_template_refresh_asset_collection(asset_library, *assets_dataptr, assets_propname);

  AssetViewListData *list_data = (AssetViewListData *)MEM_mallocN(sizeof(*list_data),
                                                                  "AssetViewListData");
  list_data->asset_library = asset_library;
  list_data->screen = CTX_wm_screen(C);

  /* TODO can we have some kind of model-view API to handle referencing, filtering and lazy loading
   * (of previews) of the items? */
  uiList *list = uiTemplateList_ex(col,
                                   C,
                                   "UI_UL_asset_view",
                                   list_id,
                                   assets_dataptr,
                                   assets_propname,
                                   active_dataptr,
                                   active_propname,
                                   nullptr,
                                   0,
                                   0,
                                   UILST_LAYOUT_BIG_PREVIEW_GRID,
                                   0,
                                   false,
                                   false,
                                   list_data);

  list->custom_activate_opname = activate_opname;
  list->custom_drag_opname = drag_opname;

  if (!list) {
    /* List creation failed. */
    MEM_freeN(list_data);
  }
}
