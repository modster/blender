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

#include <iomanip>
#include <sstream>

#include "spreadsheet_drawers.hh"

#include "UI_interface.h"
#include "UI_resources.h"

#include "UI_interface.h"
#include "UI_resources.h"

namespace blender::ed::spreadsheet {

void TextHeaderDrawer::draw_header(const HeaderDrawParams &params) const
{
  uiBut *but = uiDefIconTextBut(params.block,
                                UI_BTYPE_LABEL,
                                0,
                                ICON_NONE,
                                text_.c_str(),
                                params.xmin,
                                params.ymin,
                                params.width,
                                params.height,
                                nullptr,
                                0,
                                0,
                                0,
                                0,
                                nullptr);
  UI_but_drawflag_disable(but, UI_BUT_TEXT_LEFT);
  UI_but_drawflag_disable(but, UI_BUT_TEXT_RIGHT);
}

void ConstantTextCellDrawer::draw_cell(const CellDrawParams &params) const
{
  uiDefIconTextBut(params.block,
                   UI_BTYPE_LABEL,
                   0,
                   ICON_NONE,
                   text_.c_str(),
                   params.xmin,
                   params.ymin,
                   params.width,
                   params.height,
                   nullptr,
                   0,
                   0,
                   0,
                   0,
                   nullptr);
}

void FloatCellDrawer::draw_cell(const CellDrawParams &params) const
{
  const float value = get_value_(params.index);
  std::stringstream ss;
  ss << std::fixed << std::setprecision(3) << value;
  const std::string value_str = ss.str();
  uiDefIconTextBut(params.block,
                   UI_BTYPE_LABEL,
                   0,
                   ICON_NONE,
                   value_str.c_str(),
                   params.xmin,
                   params.ymin,
                   params.width,
                   params.height,
                   nullptr,
                   0,
                   0,
                   0,
                   0,
                   nullptr);
}

void IntCellDrawer::draw_cell(const CellDrawParams &params) const
{
  const int value = get_value_(params.index);
  const std::string value_str = std::to_string(value);
  uiDefIconTextBut(params.block,
                   UI_BTYPE_LABEL,
                   0,
                   ICON_NONE,
                   value_str.c_str(),
                   params.xmin,
                   params.ymin,
                   params.width,
                   params.height,
                   nullptr,
                   0,
                   0,
                   0,
                   0,
                   nullptr);
}

void BoolCellDrawer::draw_cell(const CellDrawParams &params) const
{
  const bool value = get_value_(params.index);
  const int icon = value ? ICON_CHECKBOX_HLT : ICON_CHECKBOX_DEHLT;
  uiDefIconTextBut(params.block,
                   UI_BTYPE_LABEL,
                   0,
                   icon,
                   "",
                   params.xmin,
                   params.ymin,
                   params.width,
                   params.height,
                   nullptr,
                   0,
                   0,
                   0,
                   0,
                   nullptr);
}

}  // namespace blender::ed::spreadsheet
