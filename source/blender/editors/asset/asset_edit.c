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
#include "BKE_lib_id.h"

#include "DNA_ID.h"
#include "DNA_asset_types.h"

#include "UI_interface_icons.h"

#include "ED_asset.h"

bool ED_asset_make_for_id(const bContext *C, ID *id)
{
  if (id->asset_data) {
    return false;
  }

  id_fake_user_set(id);

#ifdef WITH_ASSET_REPO_INFO
  BKE_asset_repository_info_global_ensure();
#endif
  id->asset_data = BKE_asset_data_create();

  UI_icon_render_id(C, NULL, id, true, true);
  /* Store reference to the ID's preview. */
  /* XXX get rid of this? File read will be a hassle and no real need for it right now. */
  id->asset_data->preview = BKE_assetdata_preview_get_from_id(id->asset_data, id);

  return true;
}
