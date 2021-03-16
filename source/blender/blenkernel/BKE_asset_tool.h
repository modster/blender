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

struct AssetTool;
struct bNodeTree;

#ifdef __cplusplus
extern "C" {
#endif

struct AssetTool *BKE_asset_tool_new(void);
struct AssetTool *BKE_asset_tool_copy(struct AssetTool *src);
void BKE_asset_tool_free(struct AssetTool *asset_tool);

#ifdef __cplusplus
}
#endif
