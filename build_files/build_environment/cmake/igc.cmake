# SPDX-License-Identifier: GPL-2.0-or-later

# CAUTION: igc requires flex 2.6.4 this is not by default available on centos 7

unpack_only(igc_llvm)
unpack_only(igc_opencl_clang)
unpack_only(igc_vcintrinsics)
unpack_only(igc_spirv_headers)
unpack_only(igc_spirv_tools)
unpack_only(igc_spirv_translator)

if(WIN32)
  set(IGC_GENERATOR "Ninja")
  set(IGC_TARGET Windows64)
else()
  set(IGC_GENERATOR "Unix Makefiles")
  set(IGC_TARGET Linux64)
endif()

set(IGC_EXTRA_ARGS
  -DIGC_OPTION__ARCHITECTURE_TARGET=${IGC_TARGET}
  -DIGC_OPTION__ARCHITECTURE_HOST=${IGC_TARGET}
)

ExternalProject_Add(external_igc
  URL file://${PACKAGE_DIR}/${IGC_FILE}
  DOWNLOAD_DIR ${DOWNLOAD_DIR}
  URL_HASH ${IGC_HASH_TYPE}=${IGC_HASH}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${LIBDIR}/igc ${DEFAULT_CMAKE_FLAGS} ${IGC_EXTRA_ARGS}

  #
  # The patch logic is.... fragile... the patches live inside external_igc but
  # are applied to projects outside its base folder, leading into the situation
  # you have to remove all external igc_ folders if you rebuild igc or it'll try
  # to reapply all patches... and fail...
  #
  # Ideally we apply all patches while we unpack the deps, but due to the dependency
  # order we can't access igc source tree yet. 
  #
  # Only "neat" solution i see is copying all patches to our patch folder.
  #
  # igc is pretty set in its way where sub projects ought to live, for some it offers
  # hooks to supply alternatives folders, other are just hardocded with no way to configure
  # we symlink everything here, since it's less work than trying to convince the cmake
  # scripts to accept alternative locations.
  #
  PATCH_COMMAND ${CMAKE_COMMAND} -E create_symlink ${BUILD_DIR}/igc_llvm/src/external_igc_llvm/ ${BUILD_DIR}/igc/src/llvm-project &&
    ${CMAKE_COMMAND} -E create_symlink ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/ ${BUILD_DIR}/igc/src/llvm-project/llvm/projects/opencl-clang &&
    ${CMAKE_COMMAND} -E create_symlink ${BUILD_DIR}/igc_spirv_translator/src/external_igc_spirv_translator/ ${BUILD_DIR}/igc/src/llvm-project/llvm/projects/llvm-spirv &&
    ${CMAKE_COMMAND} -E create_symlink ${BUILD_DIR}/igc_spirv_tools/src/external_igc_spirv_tools/ ${BUILD_DIR}/igc/src/SPIRV-Tools &&
    ${CMAKE_COMMAND} -E create_symlink ${BUILD_DIR}/igc_spirv_headers/src/external_igc_spirv_headers/ ${BUILD_DIR}/igc/src/SPIRV-Headers &&
    ${CMAKE_COMMAND} -E create_symlink ${BUILD_DIR}/igc_vcintrinsics/src/external_igc_vcintrinsics/ ${BUILD_DIR}/igc/src/vc-intrinsics &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/ < ${PATCH_DIR}/igc_opencl_clang.diff &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc/src/llvm-project/ < ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/patches/clang/0001-OpenCL-3.0-support.patch &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc/src/llvm-project/ < ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/patches/clang/0002-Remove-__IMAGE_SUPPORT__-macro-for-SPIR.patch &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc/src/llvm-project/ < ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/patches/clang/0003-Avoid-calling-ParseCommandLineOptions-in-BackendUtil.patch &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc/src/llvm-project/ < ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/patches/clang/0004-OpenCL-support-cl_ext_float_atomics.patch &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc/src/llvm-project/ < ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/patches/clang/0005-OpenCL-Add-cl_khr_integer_dot_product.patch &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc/src/llvm-project/ < ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/patches/llvm/0001-Memory-leak-fix-for-Managed-Static-Mutex.patch &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc/src/llvm-project/ < ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/patches/llvm/0002-Remove-repo-name-in-LLVM-IR.patch &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc/src/llvm-project/llvm/projects/llvm-spirv < ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/patches/spirv/0001-update-SPIR-V-headers-for-SPV_INTEL_split_barrier.patch &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc/src/llvm-project/llvm/projects/llvm-spirv < ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/patches/spirv/0002-Add-support-for-split-barriers-extension-SPV_INTEL_s.patch &&
    PATCH_COMMAND ${PATCH_CMD} -p 1 -d ${BUILD_DIR}/igc/src/llvm-project/llvm/projects/llvm-spirv < ${BUILD_DIR}/igc_opencl_clang/src/external_igc_opencl_clang/patches/spirv/0003-Support-cl_bf16_conversions.patch
  PREFIX ${BUILD_DIR}/igc
  INSTALL_DIR ${LIBDIR}/igc
  CMAKE_GENERATOR ${IGC_GENERATOR}
)

add_dependencies(
  external_igc
  external_igc_vcintrinsics
  external_igc_llvm
  external_igc_opencl_clang
  external_igc_vcintrinsics
  external_igc_spirv_headers
  external_igc_spirv_tools
  external_igc_spirv_translator
)

