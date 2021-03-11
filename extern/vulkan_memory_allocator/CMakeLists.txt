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
# The Original Code is Copyright (C) 2012, Blender Foundation
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

# avoid noisy warnings
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
  # This is not working! Find a way to suppress -Wnullability-completeness warnings
  remove_cc_flag(
    "-Wnullability-completeness"
  )
endif()

set(INC
  .
)

set(INC_SYS
  ${Vulkan_INCLUDE_DIRS}
)

set(SRC
  vk_mem_alloc_impl.cc

  vk_mem_alloc.h
)

blender_add_lib(extern_vulkan_memory_allocator "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")
