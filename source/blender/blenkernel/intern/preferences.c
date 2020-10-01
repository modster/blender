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
 * \ingroup bke
 *
 * User defined menu API.
 */

#include <string.h>

#include "MEM_guardedalloc.h"

#include "BLI_listbase.h"
#include "BLI_path_util.h"
#include "BLI_string.h"

#include "BKE_appdir.h"
#include "BKE_preferences.h"

#include "BLT_translation.h"

#include "DNA_userdef_types.h"

#define U BLI_STATIC_ASSERT(false, "Global 'U' not allowed, only use arguments passed in!")

/* -------------------------------------------------------------------- */
/** \name Asset Repositories
 * \{ */

bUserAssetRepository *BKE_preferences_asset_repository_add(UserDef *userdef,
                                                           const char *name,
                                                           const char *path)
{
  bUserAssetRepository *repository = MEM_callocN(sizeof(*repository), "bUserAssetRepository");

  BLI_addtail(&userdef->asset_repositories, repository);

  if (name) {
    BLI_strncpy(repository->name, IFACE_("Default"), sizeof(repository->name));
  }
  if (path) {
    BLI_strncpy(repository->path, path, sizeof(repository->path));
  }

  return repository;
}

/**
 * Unlink and free a repository preference member.
 * \note Free's \a repository itself.
 */
void BKE_preferences_asset_repository_remove(UserDef *userdef, bUserAssetRepository *repository)
{
  BLI_freelinkN(&userdef->asset_repositories, repository);
}

void BKE_preferences_asset_repository_default_add(UserDef *userdef)
{
  const char *asset_blend_name = "assets.blend";
  const char *doc_path = BKE_appdir_folder_default();

  /* No home or documents path found, not much we can do. */
  if (!doc_path || !doc_path[0]) {
    return;
  }

  /* Add new "Default" repository under '[doc_path]/assets.blend'. */

  bUserAssetRepository *repository = BKE_preferences_asset_repository_add(
      userdef, DATA_("Default"), NULL);
  BLI_join_dirfile(repository->path, sizeof(repository->path), doc_path, asset_blend_name);
}

/** \} */
