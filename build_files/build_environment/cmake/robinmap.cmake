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
# ***** END GPL LICENSE BLOCK *****

set(ROBINMAP_EXTRA_ARGS
)

ExternalProject_Add(external_robinmap
  URL file://${PACKAGE_DIR}/${ROBINMAP_FILE}
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  URL_HASH ${ROBINMAP_HASH_TYPE}=${ROBINMAP_HASH}
  PREFIX ${BUILD_DIR}/robinmap
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${LIBDIR}/robinmap ${DEFAULT_CMAKE_FLAGS} ${ROBINMAP_EXTRA_ARGS}
  INSTALL_DIR ${LIBDIR}/robinmap
)

if(WIN32)
  if(BUILD_MODE STREQUAL Release)
    ExternalProject_Add_Step(external_robinmap after_install
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBDIR}/zstd/include/ ${HARVEST_TARGET}/zstd/include/
      DEPENDEES install
    )
  endif()
endif()
