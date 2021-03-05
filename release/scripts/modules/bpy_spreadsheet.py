# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

import bpy

class SpreadsheetDrawer:
    def __init__(self):
        self.objects = list(bpy.context.selected_objects)

    def get_row_amount(self):
        return len(self.objects)

    def get_column_amount(self):
        return 2

    def get_top_row_cell(self, column_index):
        return ("Name", "Pass Index")[column_index]

    def get_left_column_cell(self, row_index):
        return row_index

    def get_content_cell(self, row_index, column_index):
        ob = self.objects[row_index]
        if column_index == 0:
            return (ob, "name")
        elif column_index == 1:
            return (ob, "pass_index")


def get_spreadsheet_drawer(spreadsheet_space: bpy.types.SpaceSpreadsheet):
    return SpreadsheetDrawer()
