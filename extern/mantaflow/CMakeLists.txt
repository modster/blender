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
# The Original Code is Copyright (C) 2016, Blender Foundation
# All rights reserved.
#
# The Original Code is: all of this file.
#
# Contributor(s): Sebastian Barschkis (sebbas)
#
# ***** END GPL LICENSE BLOCK *****

# Mantaflow triggers a clang-cl compiler error with versions before 9.0.1
# Since mantaflow does not appear to be using OpenMP at this point in time,
# disable the flag for now. See https://bugs.llvm.org/show_bug.cgi?id=43175 for details.

if(MSVC_CLANG AND WITH_OPENMP AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0.1")
  remove_cc_flag("-fopenmp")
endif()

# Exporting functions from the blender binary gives linker warnings on Apple arm64 systems.
# Silence them here.
if(APPLE AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "arm64"))
  if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    string(APPEND CMAKE_C_FLAGS " -fvisibility=hidden")
    string(APPEND CMAKE_CXX_FLAGS " -fvisibility=hidden")
  endif()
endif()

set(MANTAVERSION "0.13")

add_definitions(-DWITH_FLUID=1)

# Compile Mantaflow dependencies too (e.g. cnpy for numpy file IO).
# Make sure that dependencies exist before enabling this option by updating the source files in extern/
set(WITH_MANTA_DEPENDENCIES 0)

# Enable Mantaflow numpy support
set(WITH_MANTA_NUMPY 0)

if(NOT WITH_MANTA_DEPENDENCIES)
  add_definitions(-DNO_CNPY=1)
endif()

if(WITH_MANTA_DEPENDENCIES)
  set(MANTA_DEP
    dependencies
  )
endif()

if(WITH_OPENMP)
  add_definitions(-DOPENMP=1)

  if(WITH_OPENMP_STATIC)
    list(APPEND LIB
      ${OpenMP_LIBRARIES}
    )
  endif()
endif()

if(WITH_OPENMP)
  add_definitions(-DOPENMP_OFFLOAD=1)
endif()

if(WITH_OPENVDB)
  add_definitions(-DOPENVDB=1)
  # OpenVDB headers use deprecated TBB headers, silence warning.
  #add_definitions(-DTBB_SUPPRESS_DEPRECATED_MESSAGES=1)
endif()

if(WITH_OPENVDB_BLOSC)
  add_definitions(-DOPENVDB_BLOSC=1)
endif()

if(WIN32)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

if(WITH_MANTA_NUMPY AND WITH_PYTHON_NUMPY)
  add_definitions(-DNUMPY=1)
endif()

set(INC
  preprocessed
  preprocessed/fileio
  preprocessed/python
  preprocessed/plugin
  helper/pwrapper
  helper/util
)

if(WITH_MANTA_DEPENDENCIES)
  list(APPEND INC
    ${MANTA_DEP}/cnpy
  )
endif()

set(INC_SYS
  ${PYTHON_INCLUDE_DIRS}
  ${ZLIB_INCLUDE_DIRS}
)

if(WITH_MANTA_NUMPY AND WITH_PYTHON_NUMPY)
  list(APPEND INC_SYS
    ${PYTHON_NUMPY_INCLUDE_DIRS}
  )
endif()

if(WITH_OPENVDB)
  list(APPEND INC_SYS
    ${OPENVDB_INCLUDE_DIRS}
  )
  list(APPEND LIB
    ${OPENVDB_LIBRARIES}
  )
  if(WIN32)
    # OpenVDB emits lots of these, they should be suppressed through other
    # means but MSVC 16.8/16.9 has broken this functionality, so C4251 is
    # suppressed here explicitly. See
    # https://developercommunity.visualstudio.com/content/problem/1167590/bug.html
    # for details.
    string(APPEND CMAKE_CXX_FLAGS " /wd4251")
  endif()
endif()

set(SRC
  preprocessed/commonkernels.h
  preprocessed/commonkernels.h.reg.cpp
  preprocessed/conjugategrad.cpp
  preprocessed/conjugategrad.h
  preprocessed/conjugategrad.h.reg.cpp
  preprocessed/edgecollapse.cpp
  preprocessed/edgecollapse.h
  preprocessed/edgecollapse.h.reg.cpp
  preprocessed/fastmarch.cpp
  preprocessed/fastmarch.h
  preprocessed/fastmarch.h.reg.cpp
  preprocessed/fileio/iogrids.cpp
  preprocessed/fileio/iomeshes.cpp
  preprocessed/fileio/ioparticles.cpp
  preprocessed/fileio/ioutil.cpp
  preprocessed/fileio/iovdb.cpp
  preprocessed/fileio/mantaio.cpp
  preprocessed/fileio/mantaio.h
  preprocessed/fileio/mantaio.h.reg.cpp
  preprocessed/fluidsolver.cpp
  preprocessed/fluidsolver.h
  preprocessed/fluidsolver.h.reg.cpp
  preprocessed/general.cpp
  preprocessed/general.h
  preprocessed/general.h.reg.cpp
  preprocessed/gitinfo.h
  preprocessed/grid.cpp
  preprocessed/grid.h
  preprocessed/grid.h.reg.cpp
  preprocessed/grid4d.cpp
  preprocessed/grid4d.h
  preprocessed/grid4d.h.reg.cpp
  preprocessed/kernel.cpp
  preprocessed/kernel.h
  preprocessed/kernel.h.reg.cpp
  preprocessed/levelset.cpp
  preprocessed/levelset.h
  preprocessed/levelset.h.reg.cpp
  preprocessed/mesh.cpp
  preprocessed/mesh.h
  preprocessed/mesh.h.reg.cpp
  preprocessed/movingobs.cpp
  preprocessed/movingobs.h
  preprocessed/movingobs.h.reg.cpp
  preprocessed/multigrid.cpp
  preprocessed/multigrid.h
  preprocessed/multigrid.h.reg.cpp
  preprocessed/noisefield.cpp
  preprocessed/noisefield.h
  preprocessed/noisefield.h.reg.cpp
  preprocessed/particle.cpp
  preprocessed/particle.h
  preprocessed/particle.h.reg.cpp
  preprocessed/plugin/advection.cpp
  preprocessed/plugin/apic.cpp
  preprocessed/plugin/extforces.cpp
  preprocessed/plugin/fire.cpp
  preprocessed/plugin/flip.cpp
  preprocessed/plugin/fluidguiding.cpp
  preprocessed/plugin/initplugins.cpp
  preprocessed/plugin/kepsilon.cpp
  preprocessed/plugin/meshplugins.cpp
  preprocessed/plugin/pressure.cpp
  preprocessed/plugin/ptsplugins.cpp
  preprocessed/plugin/secondaryparticles.cpp
  preprocessed/plugin/surfaceturbulence.cpp
#  preprocessed/plugin/viscosity.cpp
  preprocessed/plugin/vortexplugins.cpp
  preprocessed/plugin/waveletturbulence.cpp
  preprocessed/plugin/waves.cpp
  preprocessed/python/defines.py
  preprocessed/python/defines.py.reg.cpp
  preprocessed/registration.cpp
  preprocessed/shapes.cpp
  preprocessed/shapes.h
  preprocessed/shapes.h.reg.cpp
  preprocessed/test.cpp
  preprocessed/timing.cpp
  preprocessed/timing.h
  preprocessed/timing.h.reg.cpp
  preprocessed/turbulencepart.cpp
  preprocessed/turbulencepart.h
  preprocessed/turbulencepart.h.reg.cpp
  preprocessed/vortexpart.cpp
  preprocessed/vortexpart.h
  preprocessed/vortexpart.h.reg.cpp
  preprocessed/vortexsheet.cpp
  preprocessed/vortexsheet.h
  preprocessed/vortexsheet.h.reg.cpp

  helper/pwrapper/manta.h
  helper/pwrapper/pclass.cpp
  helper/pwrapper/pclass.h
  helper/pwrapper/pconvert.cpp
  helper/pwrapper/pconvert.h
  helper/pwrapper/pvec3.cpp
  helper/pwrapper/pythonInclude.h
  helper/pwrapper/registry.cpp
  helper/pwrapper/registry.h
  helper/util/integrator.h
  helper/util/interpol.h
  helper/util/interpolHigh.h
  helper/util/matrixbase.h
  helper/util/mcubes.h
  helper/util/quaternion.h
  helper/util/randomstream.h
  helper/util/rcmatrix.h
  helper/util/simpleimage.cpp
  helper/util/simpleimage.h
  helper/util/solvana.h
  helper/util/vector4d.cpp
  helper/util/vector4d.h
  helper/util/vectorbase.cpp
  helper/util/vectorbase.h
)

if(WITH_MANTA_DEPENDENCIES)
  list(APPEND SRC
    ${MANTA_DEP}/cnpy/cnpy.cpp
    ${MANTA_DEP}/cnpy/cnpy.h
  )
endif()
if(WITH_MANTA_NUMPY AND WITH_PYTHON_NUMPY)
  list(APPEND SRC
    preprocessed/plugin/numpyconvert.cpp
    preprocessed/plugin/tfplugins.cpp
    helper/pwrapper/numpyWrap.cpp
    helper/pwrapper/numpyWrap.h
  )
endif()

set(LIB
  ${PYTHON_LINKFLAGS}
  ${PYTHON_LIBRARIES}
  ${OPENVDB_LIBRARIES}  
)

#blender_add_lib(extern_mantaflow "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")

add_library(extern_mantaflow SHARED "${SRC}")

include_directories(preprocessed)
include_directories(preprocessed/fileio)
include_directories(preprocessed/python)
include_directories(preprocessed/plugin)
include_directories(helper/pwrapper)
include_directories(helper/util)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${OPENVDB_INCLUDE_DIRS})

#target_include_directories(extern_mantaflow PRIVATE "${INC}")
target_link_libraries(extern_mantaflow PRIVATE "${LIB}")

blender_source_group(extern_mantaflow "${SRC}")
