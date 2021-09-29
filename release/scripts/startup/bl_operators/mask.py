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

# <pep8 compliant>
import bpy
from bpy.types import Operator
from bpy.props import EnumProperty


class MASK_OT_draw_mask(Operator):
    """Smart code to draw a mask"""
    bl_label = "Draw a mask"
    bl_idname = "mask.draw_mask"

    @classmethod
    def poll(cls, context):
        if context.space_data.type == 'CLIP_EDITOR':
            clip = context.space_data.clip
            return clip
        else:
            return True

    type: EnumProperty(
        name="Type",
        items=(
	        ('AUTO', "Auto", ""),
	        ('VECTOR', "Vector", ""),
	        ('ALIGNED', "Aligned Single", ""),
	        ('ALIGNED_DOUBLESIDE', "Aligned", ""),
	        ('FREE', "Free", ""),
        ),
    )

    def execute(self, context):
        return bpy.ops.mask.add_vertex_slide(
            MASK_OT_add_vertex={
                "type": self.type,
            }
        )

    def invoke(self, context, _event):
        bpy.ops.mask.add_vertex_slide(
            'INVOKE_REGION_WIN',
            MASK_OT_add_vertex={
                "type": self.type,
            }
        )

        # ignore return from operators above because they are 'RUNNING_MODAL',
        # and cause this one not to be freed. T24671.
        return {'FINISHED'}


classes = (
    MASK_OT_draw_mask,
)
