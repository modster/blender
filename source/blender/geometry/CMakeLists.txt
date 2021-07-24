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
        .
        ../blenkernel
        ../blenlib
        ../makesdna
        ../../../intern/guardedalloc
        )

set(SRC
        intern/remesh_blocks.c
        GEO_mesh_remesh_blocks.h
        )

set(LIB
        bf_blenkernel
        bf_blenlib
        )

if(WITH_REMESH_DUALCON)
    list(APPEND INC
            ../../../intern/dualcon
            )
    list(APPEND LIB
            bf_intern_dualcon
            )
    add_definitions(-DWITH_REMESH_DUALCON)
endif()

blender_add_lib(bf_geometry "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")
