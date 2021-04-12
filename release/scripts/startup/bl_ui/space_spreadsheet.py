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

def get_visible_column_data_type(columns, column_name):
    for column in columns:
        if column.id.name == column_name:
            return column.data_type
    return 'FLOAT' # Show float data type by default, arbitrary choice.

def get_operation_string(operation):
    if operation == 'EQUAL':
        return "=="
    elif operation == 'GREATER':
        return  ">"
    elif operation == 'LESS':
        return "<"
    return ""

def get_value_string(filter, data_type):
    if data_type == 'FLOAT':
        return "%.3f" % filter.value_float
    elif data_type == 'INT32':
        return str(filter.value_int)
    elif data_type == 'BOOLEAN':
        return "True" if filter.value_boolean else "False"
    return ""

def get_filter_label_text(filter, column_name, operation, data_type):
    if len(column_name) == 0:
        return "Filter"

    return "%s %s %s" % (column_name, 
                         get_operation_string(operation), 
                         get_value_string(filter, data_type))
    

def draw_filter(layout, filter, columns, index):
    box = layout.box()

    column_name = filter.column_name
    operation = filter.operation
    data_type = get_visible_column_data_type(columns, column_name)
    label_text = get_filter_label_text(filter, column_name, operation, data_type)

    row = box.row(align=True)
    row.prop(filter, "show_expanded", text="", emboss=False)
    row.prop(filter, "enabled", text="")
    row.label(text=label_text)
    sub = row.row()
    sub.alignment = 'RIGHT'
    sub.emboss = 'NONE'
    sub.operator("spreadsheet.remove_rule", text="", icon='X').index = index

    if not filter.show_expanded:
        return
    
    box.prop(filter, "column_name", text="")
    if data_type != 'BOOLEAN':
        box.prop(filter, "operation", text="")

    if data_type == 'FLOAT':
        box.prop(filter, "value_float", text="Value")
        if operation == 'EQUAL':
            box.prop(filter, "threshold")
    elif data_type == 'INT32':
        box.prop(filter, "value_int", text="Value")
    elif data_type == 'BOOLEAN':
        box.prop(filter, "value_boolean", text="Value")


class SPREADSHEET_PT_filter_rules(bpy.types.Panel):
    bl_space_type = 'SPREADSHEET'
    bl_region_type = 'HEADER'
    bl_label = "Filters"
    bl_parent_id = 'SPREADSHEET_PT_filter'

    def draw_header(self, context):
        layout = self.layout
        layout.label(text="Filters")
        sub = layout.row()
        sub.alignment = 'RIGHT'
        sub.emboss = 'NONE'
        sub.operator("spreadsheet.add_rule", text="", icon='ADD')

    def draw(self, context):
        layout = self.layout
        space = context.space_data

        index = 0
        for filter in space.row_filters:
            draw_filter(layout, filter, space.columns, index)
            index += 1


class SPREADSHEET_PT_filter(bpy.types.Panel):
    bl_space_type = 'SPREADSHEET'
    bl_region_type = 'HEADER'
    bl_label = "Filter"

    def draw(self, context):
        layout = self.layout
        space = context.space_data

        pinned_id = space.pinned_id
        used_id = pinned_id if pinned_id else context.active_object

        if isinstance(used_id, bpy.types.Object) and used_id.mode == 'EDIT':
            layout.prop(space, "show_only_selected", text="Selected Only")


class SPREADSHEET_HT_header(bpy.types.Header):
    bl_space_type = 'SPREADSHEET'

    def draw(self, context):
        layout = self.layout
        space = context.space_data

        layout.template_header()

        pinned_id = space.pinned_id
        used_id = pinned_id if pinned_id else context.active_object

        layout.prop(space, "object_eval_state", text="")
        if space.object_eval_state != 'ORIGINAL':
            layout.prop(space, "geometry_component_type", text="")
        if space.geometry_component_type != 'INSTANCES':
            layout.prop(space, "attribute_domain", text="")

        if used_id:
            layout.label(text=used_id.name, icon='OBJECT_DATA')

        layout.operator("spreadsheet.toggle_pin", text="", icon='PINNED' if pinned_id else 'UNPINNED', emboss=False)

        layout.separator_spacer()

        row = layout.row(align=True)
        row.prop(space, "use_filter", toggle=True, icon='FILTER', icon_only=True)
        row.popover("SPREADSHEET_PT_filter", text="")


classes = (
    SPREADSHEET_HT_header,
    SPREADSHEET_PT_filter,
    SPREADSHEET_PT_filter_rules,
)

if __name__ == "__main__":  # Only for live edit.
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
