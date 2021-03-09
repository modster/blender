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
 * \ingroup editors
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct AssetLibraryReference;

bool ED_asset_mark_id(const struct bContext *C, struct ID *id);
bool ED_asset_clear_id(struct ID *id);

bool ED_asset_can_make_single_from_context(const struct bContext *C);

int ED_asset_library_reference_to_enum_value(const struct AssetLibraryReference *library);
AssetLibraryReference ED_asset_library_reference_from_enum_value(int value);

void ED_assetlist_fetch(const struct AssetLibraryReference *library_reference,
                        const struct AssetFilterSettings *filter_settings,
                        const bContext *C);
void ED_assetlist_ensure_previews_job(const struct AssetLibraryReference *library_reference,
                                      bContext *C);
void ED_assetlist_storage_exit(void);

struct FileDirEntry;
struct ImBuf *ED_assetlist_asset_image_get(const struct FileDirEntry *file);
const char *ED_assetlist_library_path(const AssetLibraryReference *library_reference);

void ED_operatortypes_asset(void);

#ifdef __cplusplus
}
#endif

/* TODO move to C++ asset-list header? */
#ifdef __cplusplus
#  include "BLI_function_ref.hh"
/* Can return false to stop iterating. */
using AssetListIterFn = blender::FunctionRef<bool(FileDirEntry &)>;
void ED_assetlist_iterate(const AssetLibraryReference *library_reference, AssetListIterFn fn);
#endif
