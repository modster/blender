# SPDX-License-Identifier: GPL-2.0-or-later

set(DPCPP_EXTRA_ARGS
)

ExternalProject_Add(external_dpcpp
  URL file://${PACKAGE_DIR}/${DPCPP_FILE}
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  URL_HASH ${DPCPP_HASH_TYPE}=${DPCPP_HASH}
  PREFIX ${BUILD_DIR}/dpcpp
  CONFIGURE_COMMAND ${PYTHON_BINARY} ${BUILD_DIR}/dpcpp/src/external_dpcpp/buildbot/configure.py
  BUILD_COMMAND ${PYTHON_BINARY} ${BUILD_DIR}/dpcpp/src/external_dpcpp/buildbot/compile.py
  INSTALL_COMMAND echo "."
  INSTALL_DIR ${LIBDIR}/dpcpp
)

add_dependencies(
  external_dpcpp
  external_python
  external_python_site_packages
)
