# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ***** END GPL LICENSE BLOCK *****

set(INC
  ../include
  ../../blenfont
  ../../blenkernel
  ../../blenlib
  ../../blenloader
  ../../blentranslation
  ../../depsgraph
  ../../draw
  ../../gpu
  ../../imbuf
  ../../makesdna
  ../../makesrna
  ../../python
  ../../render
  ../../windowmanager
  ../../../../intern/glew-mx
  ../../../../intern/guardedalloc
)

set(SRC
  interface.cc
  interface_align.c
  interface_anim.c
  interface_button_group.cc
  interface_context_menu.cc
  interface_draw.cc
  interface_eyedropper.c
  interface_eyedropper_color.c
  interface_eyedropper_colorband.c
  interface_eyedropper_datablock.c
  interface_eyedropper_depth.c
  interface_eyedropper_driver.c
  interface_eyedropper_gpencil_color.c
  interface_handlers.cc
  interface_icons.c
  interface_icons_event.cc
  interface_layout.cc
  interface_ops.c
  interface_panel.cc
  interface_query.cc
  interface_region_color_picker.cc
  interface_region_hud.cc
  interface_region_menu_pie.c
  interface_region_menu_popup.c
  interface_region_popover.c
  interface_region_popup.cc
  interface_region_search.cc
  interface_region_tooltip.c
  interface_regions.cc
  interface_style.cc
  interface_template_search_menu.cc
  interface_template_search_operator.cc
  interface_templates.cc
  interface_undo.cc
  interface_utils.cc
  interface_widgets.c
  resources.cc
  view2d.c
  view2d_draw.c
  view2d_gizmo_navigate.c
  view2d_ops.c

  interface_eyedropper_intern.h
  interface_intern.h
  interface_regions_intern.h
)

set(LIB
  bf_blenkernel
  bf_blenlib
  bf_editor_datafiles
)

if(WITH_INTERNATIONAL)
  add_definitions(-DWITH_INTERNATIONAL)
endif()

if(WITH_HEADLESS)
  add_definitions(-DWITH_HEADLESS)
endif()

if(WITH_PYTHON)
  add_definitions(-DWITH_PYTHON)
endif()

if(WIN32)
  if(WITH_INPUT_IME)
    add_definitions(-DWITH_INPUT_IME)
  endif()
endif()


blender_add_lib(bf_editor_interface "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")
