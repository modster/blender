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
 */

#include "BKE_asset_tool.h"

#include "DNA_node_types.h"

#include "MEM_guardedalloc.h"

#include "BLI_listbase.h"

AssetTool *BKE_asset_tool_new()
{
  AssetTool *asset_tool = (AssetTool *)MEM_callocN(sizeof(AssetTool), __func__);
  return asset_tool;
}

AssetTool *BKE_asset_tool_copy(AssetTool *src)
{
  AssetTool *asset_tool = (AssetTool *)MEM_dupallocN(src);
  return asset_tool;
}

void BKE_asset_tool_free(AssetTool *asset_tool)
{
  MEM_freeN(asset_tool);
}
