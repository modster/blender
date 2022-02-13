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


set(PYSTRING_EXTRA_ARGS
)

ExternalProject_Add(external_pystring
  URL file://${PACKAGE_DIR}/${PYSTRING_FILE}
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  URL_HASH ${PYSTRING_HASH_TYPE}=${PYSTRING_HASH}
  PREFIX ${BUILD_DIR}/pystring
  PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${PATCH_DIR}/cmakelists_pystring.txt ${BUILD_DIR}/pystring/src/external_pystring/CMakeLists.txt
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${LIBDIR}/pystring ${DEFAULT_CMAKE_FLAGS} ${PYSTRING_EXTRA_ARGS}
  INSTALL_DIR ${LIBDIR}/pystring
)

if(WIN32)
  ExternalProject_Add_Step(external_pystring after_install
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBDIR}/pystring/lib ${HARVEST_TARGET}/pystring/lib
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBDIR}/pystring/include ${HARVEST_TARGET}/pystring/include
    DEPENDEES install
  )
endif()
