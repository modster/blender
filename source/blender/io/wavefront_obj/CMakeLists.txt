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
# The Original Code is Copyright (C) 2020, Blender Foundation
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

set(INC
  .
  ./exporter
  ./importer
  ../../blenkernel
  ../../blenlib
  ../../bmesh
  ../../bmesh/intern
  ../../depsgraph
  ../../editors/include
  ../../makesdna
  ../../makesrna
  ../../nodes
  ../../windowmanager
  ../../../../intern/guardedalloc
)

set(INC_SYS

)

set(SRC
  IO_wavefront_obj.cc
  exporter/obj_exporter.cc
  exporter/obj_export_file_writer.cc
  exporter/obj_export_mesh.cc
  exporter/obj_export_mtl.cc
  exporter/obj_export_nurbs.cc
  importer/importer_mesh_utils.cc
  importer/obj_import_file_reader.cc
  importer/obj_importer.cc
  importer/obj_import_mesh.cc
  importer/obj_import_mtl.cc
  importer/obj_import_nurbs.cc
  importer/obj_import_objects.cc
  importer/parser_string_utils.cc

  IO_wavefront_obj.h
  exporter/obj_exporter.hh
  exporter/obj_export_file_writer.hh
  exporter/obj_export_io.hh
  exporter/obj_export_mesh.hh
  exporter/obj_export_mtl.hh
  exporter/obj_export_nurbs.hh
  importer/importer_mesh_utils.hh
  importer/obj_import_file_reader.hh
  importer/obj_importer.hh
  importer/obj_import_mesh.hh
  importer/obj_import_mtl.hh
  importer/obj_import_nurbs.hh
  importer/obj_import_objects.hh
  importer/parser_string_utils.hh
)

set(LIB
  bf_blenkernel
)

blender_add_lib(bf_wavefront_obj "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")

if(WITH_GTESTS)
  set(TEST_SRC
    tests/obj_exporter_tests.cc
    tests/obj_exporter_tests.hh
  )

  set(TEST_INC
    ${INC}

    ../../blenloader
    ../../../../tests/gtests
  )

  set(TEST_LIB
    ${LIB}

    bf_blenloader_tests
    bf_wavefront_obj
  )

  include(GTestTesting)
  blender_add_test_lib(bf_wavefront_obj_tests "${TEST_SRC}" "${TEST_INC}" "${INC_SYS}" "${TEST_LIB}")
  add_dependencies(bf_wavefront_obj_tests bf_wavefront_obj)
endif()
