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


set(IMATH_EXTRA_ARGS
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_TESTING=OFF
)

ExternalProject_Add(external_imath
  URL file://${PACKAGE_DIR}/${IMATH_FILE}
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  URL_HASH ${IMATH_HASH_TYPE}=${IMATH_HASH}
  PREFIX ${BUILD_DIR}/imath
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${LIBDIR}/imath ${DEFAULT_CMAKE_FLAGS} ${IMATH_EXTRA_ARGS}
  INSTALL_DIR ${LIBDIR}/imath
)

if(WIN32)
  ExternalProject_Add_Step(external_imath after_install
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBDIR}/imath/lib ${HARVEST_TARGET}/imath/lib
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBDIR}/imath/include ${HARVEST_TARGET}/imath/include
    DEPENDEES install
  )
endif()
