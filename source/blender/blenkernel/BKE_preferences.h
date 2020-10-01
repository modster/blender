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

#pragma once

/** \file
 * \ingroup bke
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "BLI_compiler_attrs.h"

struct UserDef;
struct bUserAssetRepository;

void BKE_preferences_asset_repository_free(struct bUserAssetRepository *repository) ATTR_NONNULL();

struct bUserAssetRepository *BKE_preferences_asset_repository_add(struct UserDef *userdef,
                                                                  const char *name,
                                                                  const char *path)
    ATTR_NONNULL(1);
void BKE_preferences_asset_repository_remove(struct UserDef *userdef,
                                             struct bUserAssetRepository *repository)
    ATTR_NONNULL();

void BKE_preferences_asset_repository_default_add(struct UserDef *userdef) ATTR_NONNULL();

#ifdef __cplusplus
}
#endif
