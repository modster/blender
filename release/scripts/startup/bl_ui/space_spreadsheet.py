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

def draw_filter(layout, filter, index):
    box = layout.box()

    row = box.row(align=True)
    row.emboss = 'NONE'
    row.prop(filter, "show_expanded", text="")
    row.prop(filter, "enabled", text="")
    row.label(text=filter.column_name if len(filter.column_name) > 0 else "Rule")
    sub = row.row()
    sub.alignment = 'RIGHT'
    sub.emboss = 'NONE'
    sub.operator("spreadsheet.remove_rule", text="", icon='X').index = index


    if filter.show_expanded:
        box.prop(filter, "column_name", text="")
        box.prop(filter, "operation", text="")

        box.prop(filter, "value_float", text="Value")
        box.prop(filter, "value_int", text="Value")
        box.prop(filter, "value_color", text="")
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
            draw_filter(layout, filter, index)
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
