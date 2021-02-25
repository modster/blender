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

#include "UI_interface.h"
#include "UI_resources.h"

#include "BLF_api.h"

#include "DNA_userdef_types.h"

#include "spreadsheet_drawers.hh"
#include "spreadsheet_from_geometry.hh"

namespace blender::ed::spreadsheet {

using blender::bke::ReadAttribute;
using blender::bke::ReadAttributePtr;

void columns_from_geometry_attributes(const GeometryComponent &component,
                                      const AttributeDomain domain,
                                      ResourceCollector &resources,
                                      SpreadsheetLayout &spreadsheet_layout)
{
  Vector<std::string> attribute_names;

  component.attribute_foreach(
      [&](const StringRef attribute_name, const AttributeMetaData &UNUSED(meta_data)) {
        attribute_names.append(attribute_name);
        return true;
      });

  std::sort(attribute_names.begin(),
            attribute_names.end(),
            [](const std::string &a, const std::string &b) {
              return BLI_strcasecmp_natural(a.c_str(), b.c_str()) < 0;
            });

  const int fontid = UI_style_get()->widget.uifont_id;
  const int header_name_padding = UI_UNIT_X;
  const int minimum_column_width = 2 * UI_UNIT_X;

  auto get_column_width = [&](StringRef name) {
    const int text_width = BLF_width(fontid, name.data(), name.size());
    const int column_width = std::max(text_width + header_name_padding, minimum_column_width);
    return column_width;
  };

  for (StringRef attribute_name : attribute_names) {
    ReadAttributePtr owned_attribute = component.attribute_try_get_for_read(attribute_name);
    if (owned_attribute->domain() != domain) {
      continue;
    }
    ReadAttribute *attribute = owned_attribute.get();
    resources.add(std::move(owned_attribute), "read attribute");

    const CustomDataType data_type = attribute->custom_data_type();
    switch (data_type) {
      case CD_PROP_FLOAT: {
        HeaderDrawer &header_drawer = resources.construct<TextHeaderDrawer>(
            "attribute header drawer", attribute_name);

        FunctionRef<float(int)> get_value = resources.add_value(
            [attribute](int index) {
              float value;
              attribute->get(index, &value);
              return value;
            },
            "get float value");

        CellDrawer &cell_drawer = resources.construct<FloatCellDrawer>("float cell drawer",
                                                                       get_value);

        spreadsheet_layout.columns.append(
            {get_column_width(attribute_name), &header_drawer, &cell_drawer});
        break;
      }
      case CD_PROP_FLOAT2: {
        static std::array<char, 2> axis_char = {'X', 'Y'};
        for (const int i : IndexRange(2)) {
          std::string header_name = attribute_name + " " + axis_char[i];
          HeaderDrawer &header_drawer = resources.construct<TextHeaderDrawer>(
              "attribute header drawer", header_name);

          FunctionRef<float(int)> get_value = resources.add_value(
              [attribute, i](int index) {
                blender::float2 value;
                attribute->get(index, &value);
                return value[i];
              },
              "get float2 value");

          CellDrawer &cell_drawer = resources.construct<FloatCellDrawer>("float cell drawer",
                                                                         get_value);

          spreadsheet_layout.columns.append(
              {get_column_width(header_name), &header_drawer, &cell_drawer});
        }
        break;
      }
      case CD_PROP_FLOAT3: {
        static std::array<char, 3> axis_char = {'X', 'Y', 'Z'};
        for (const int i : IndexRange(3)) {
          std::string header_name = attribute_name + " " + axis_char[i];
          HeaderDrawer &header_drawer = resources.construct<TextHeaderDrawer>(
              "attribute header drawer", header_name);

          FunctionRef<float(int)> get_value = resources.add_value(
              [attribute, i](int index) {
                float3 value;
                attribute->get(index, &value);
                return value[i];
              },
              "get float3 value");

          CellDrawer &cell_drawer = resources.construct<FloatCellDrawer>("float cell drawer",
                                                                         get_value);

          spreadsheet_layout.columns.append(
              {get_column_width(header_name), &header_drawer, &cell_drawer});
        }
        break;
      }
      case CD_PROP_COLOR: {
        static std::array<char, 4> channel_char = {'R', 'G', 'B', 'A'};
        for (const int i : IndexRange(4)) {
          std::string header_name = attribute_name + " " + channel_char[i];
          HeaderDrawer &header_drawer = resources.construct<TextHeaderDrawer>(
              "attribute header drawer", header_name);

          FunctionRef<float(int)> get_value = resources.add_value(
              [attribute, i](int index) {
                blender::Color4f value;
                attribute->get(index, &value);
                return value[i];
              },
              "get color value");

          CellDrawer &cell_drawer = resources.construct<FloatCellDrawer>("float cell drawer",
                                                                         get_value);

          spreadsheet_layout.columns.append(
              {get_column_width(header_name), &header_drawer, &cell_drawer});
        }
        break;
      }
      case CD_PROP_INT32: {
        HeaderDrawer &header_drawer = resources.construct<TextHeaderDrawer>(
            "attribute header drawer", attribute_name);

        FunctionRef<int(int)> get_value = resources.add_value(
            [attribute](int index) {
              int value;
              attribute->get(index, &value);
              return value;
            },
            "get int value");

        CellDrawer &cell_drawer = resources.construct<IntCellDrawer>("int cell drawer", get_value);

        spreadsheet_layout.columns.append(
            {get_column_width(attribute_name), &header_drawer, &cell_drawer});
        break;
      }
      case CD_PROP_BOOL: {
        HeaderDrawer &header_drawer = resources.construct<TextHeaderDrawer>(
            "attribute header drawer", attribute_name);

        FunctionRef<bool(int)> get_value = resources.add_value(
            [attribute](int index) {
              bool value;
              attribute->get(index, &value);
              return value;
            },
            "get bool value");

        CellDrawer &cell_drawer = resources.construct<BoolCellDrawer>("bool cell drawer",
                                                                      get_value);

        spreadsheet_layout.columns.append(
            {get_column_width(attribute_name), &header_drawer, &cell_drawer});
        break;
      }
      default:
        break;
    }
  }
}

}  // namespace blender::ed::spreadsheet
