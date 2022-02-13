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

if(WIN32)
  # cmake for windows
  set(JPEG_EXTRA_ARGS
    -DNASM=${NASM_PATH}
    -DWITH_JPEG8=ON
    -DCMAKE_DEBUG_POSTFIX=d
    -DWITH_CRT_DLL=On
    -DENABLE_SHARED=OFF
    -DENABLE_STATIC=ON
  )

  ExternalProject_Add(external_jpeg
    URL file://${PACKAGE_DIR}/${JPEG_FILE}
    DOWNLOAD_DIR ${DOWNLOAD_DIR}
    URL_HASH ${JPEG_HASH_TYPE}=${JPEG_HASH}
    PREFIX ${BUILD_DIR}/jpeg
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${LIBDIR}/jpeg ${DEFAULT_CMAKE_FLAGS} ${JPEG_EXTRA_ARGS}
    INSTALL_DIR ${LIBDIR}/jpeg
  )

  if(BUILD_MODE STREQUAL Release)
    set(JPEG_LIBRARY jpeg-static${LIBEXT})
  else()
    set(JPEG_LIBRARY jpeg-staticd${LIBEXT})
  endif()

  if(BUILD_MODE STREQUAL Release)
    ExternalProject_Add_Step(external_jpeg after_install
      COMMAND ${CMAKE_COMMAND} -E copy ${LIBDIR}/jpeg/lib/${JPEG_LIBRARY}  ${LIBDIR}/jpeg/lib/jpeg${LIBEXT}
      DEPENDEES install
    )
  endif()

else(WIN32)
  # cmake for unix
  set(JPEG_EXTRA_ARGS
    -DWITH_JPEG8=ON
    -DENABLE_STATIC=ON
    -DENABLE_SHARED=OFF
    -DCMAKE_INSTALL_LIBDIR=${LIBDIR}/jpeg/lib)

  ExternalProject_Add(external_jpeg
    URL file://${PACKAGE_DIR}/${JPEG_FILE}
    DOWNLOAD_DIR ${DOWNLOAD_DIR}
    URL_HASH ${JPEG_HASH_TYPE}=${JPEG_HASH}
    PREFIX ${BUILD_DIR}/jpeg
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${LIBDIR}/jpeg ${DEFAULT_CMAKE_FLAGS} ${JPEG_EXTRA_ARGS}
    INSTALL_DIR ${LIBDIR}/jpeg
  )

  set(JPEG_LIBRARY libjpeg${LIBEXT})
endif()
