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
 #
 # The Original Code is Copyright (C) 2006, Blender Foundation
 # All rights reserved.
 # ***** END GPL LICENSE BLOCK *****

 set(INC
         .
         ../blenfont
         ../blenkernel
         ../blenlib
         ../blentranslation
         ../functions
         ../makesdna
         ../makesrna
         ../../../intern/guardedalloc
         ../makesdna/intern
         ${CMAKE_BINARY_DIR}/source/blender/makesdna/intern
         )

 set(SRC
         intern/solidify_nonmanifold.c
         GEO_solidifiy.h
         intern/remesh_blocks.c
         GEO_mesh_remesh_blocks.h
         intern/mesh_merge_by_distance.c
         intern/pointcloud_merge_by_distance.cc
         GEO_mesh_merge_by_distance.h
         GEO_pointcloud_merge_by_distance.h
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