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

#include <memory>
#include <string>

#include "BKE_asset.h"
#include "BKE_context.h"
#include "BKE_lib_id.h"
#include "BKE_report.h"

#include "BLI_utility_mixins.hh"

#include "BLO_readfile.h"

#include "DNA_ID.h"
#include "DNA_asset_types.h"
#include "DNA_space_types.h"

#include "MEM_guardedalloc.h"

#include "UI_interface_icons.h"

#include "RNA_access.h"

#include "ED_asset.h"

using namespace blender;

bool ED_asset_mark_id(const bContext *C, ID *id)
{
  if (id->asset_data) {
    return false;
  }
  if (!BKE_id_can_be_asset(id)) {
    return false;
  }

  id_fake_user_set(id);

  id->asset_data = BKE_asset_metadata_create();

  UI_icon_render_id(C, nullptr, id, ICON_SIZE_PREVIEW, true);

  /* Important for asset storage to update properly! */
  ED_assetlist_storage_tag_main_data_dirty();

  return true;
}

bool ED_asset_clear_id(ID *id)
{
  if (!id->asset_data) {
    return false;
  }
  BKE_asset_metadata_free(&id->asset_data);
  /* Don't clear fake user here, there's no guarantee that it was actually set by
   * #ED_asset_mark_id(), it might have been something/someone else. */

  /* Important for asset storage to update properly! */
  ED_assetlist_storage_tag_main_data_dirty();

  return true;
}

bool ED_asset_can_make_single_from_context(const bContext *C)
{
  /* Context needs a "id" pointer to be set for #ASSET_OT_mark()/#ASSET_OT_clear() to use. */
  return CTX_data_pointer_get_type_silent(C, "id", &RNA_ID).data != nullptr;
}

/* TODO better place? */
/* TODO What about the setter and the itemf? */
#include "BKE_preferences.h"
#include "DNA_asset_types.h"
#include "DNA_userdef_types.h"
int ED_asset_library_reference_to_enum_value(const AssetLibraryReference *library)
{
  /* Simple case: Predefined repo, just set the value. */
  if (library->type < ASSET_LIBRARY_CUSTOM) {
    return library->type;
  }

  /* Note that the path isn't checked for validity here. If an invalid library path is used, the
   * Asset Browser can give a nice hint on what's wrong. */
  const bUserAssetLibrary *user_library = BKE_preferences_asset_library_find_from_index(
      &U, library->custom_library_index);
  if (user_library) {
    return ASSET_LIBRARY_CUSTOM + library->custom_library_index;
  }

  BLI_assert(0);
  return ASSET_LIBRARY_LOCAL;
}

AssetLibraryReference ED_asset_library_reference_from_enum_value(int value)
{
  AssetLibraryReference library;

  /* Simple case: Predefined repo, just set the value. */
  if (value < ASSET_LIBRARY_CUSTOM) {
    library.type = value;
    library.custom_library_index = -1;
    BLI_assert(ELEM(value, ASSET_LIBRARY_LOCAL));
    return library;
  }

  const bUserAssetLibrary *user_library = BKE_preferences_asset_library_find_from_index(
      &U, value - ASSET_LIBRARY_CUSTOM);

  /* Note that the path isn't checked for validity here. If an invalid library path is used, the
   * Asset Browser can give a nice hint on what's wrong. */
  const bool is_valid = (user_library->name[0] && user_library->path[0]);
  if (!user_library) {
    library.type = ASSET_LIBRARY_LOCAL;
    library.custom_library_index = -1;
  }
  else if (user_library && is_valid) {
    library.custom_library_index = value - ASSET_LIBRARY_CUSTOM;
    library.type = ASSET_LIBRARY_CUSTOM;
  }
  return library;
}

class AssetTemporaryIDConsumer : NonCopyable, NonMovable {
  const AssetHandle &handle_;
  TempLibraryContext *temp_lib_context_ = nullptr;

 public:
  AssetTemporaryIDConsumer(const AssetHandle &handle) : handle_(handle)
  {
  }
  ~AssetTemporaryIDConsumer()
  {
    if (temp_lib_context_) {
      BLO_library_temp_free(temp_lib_context_);
    }
  }

  ID *get_local_id()
  {
    return ED_assetlist_asset_local_id_get(&handle_);
  }

  ID *import_id(const AssetLibraryReference &asset_library,
                ID_Type id_type,
                Main &bmain,
                ReportList &reports)
  {
    std::string asset_path = ED_assetlist_asset_filepath_get(asset_library, handle_);
    if (asset_path.empty()) {
      return nullptr;
    }

    char blend_file_path[FILE_MAX_LIBEXTRA];
    char *group = NULL;
    char *asset_name = NULL;
    BLO_library_path_explode(asset_path.c_str(), blend_file_path, &group, &asset_name);

    temp_lib_context_ = BLO_library_temp_load_id(
        &bmain, blend_file_path, id_type, asset_name, &reports);

    if (temp_lib_context_ == nullptr || temp_lib_context_->temp_id == nullptr) {
      BKE_reportf(&reports, RPT_ERROR, "Unable to load %s from %s", asset_name, blend_file_path);
      return nullptr;
    }

    BLI_assert(GS(temp_lib_context_->temp_id->name) == id_type);
    return temp_lib_context_->temp_id;
  }
};

AssetTempIDConsumer *ED_asset_temp_id_consumer_create(const AssetHandle *handle)
{
  if (!handle) {
    return nullptr;
  }
  return reinterpret_cast<AssetTempIDConsumer *>(
      OBJECT_GUARDED_NEW(AssetTemporaryIDConsumer, *handle));
}

void ED_asset_temp_id_consumer_free(AssetTempIDConsumer **consumer)
{
  OBJECT_GUARDED_SAFE_DELETE(*consumer, AssetTemporaryIDConsumer);
}

ID *ED_asset_temp_id_consumer_ensure_local_id(AssetTempIDConsumer *consumer_,
                                              const AssetLibraryReference *asset_library,
                                              ID_Type id_type,
                                              Main *bmain,
                                              ReportList *reports)
{
  if (!(consumer_ && asset_library && bmain && reports)) {
    return nullptr;
  }
  AssetTemporaryIDConsumer *consumer = reinterpret_cast<AssetTemporaryIDConsumer *>(consumer_);

  if (ID *local_id = consumer->get_local_id()) {
    return local_id;
  }
  return consumer->import_id(*asset_library, id_type, *bmain, *reports);
}
