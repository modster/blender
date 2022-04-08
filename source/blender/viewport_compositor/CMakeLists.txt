	# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright 2022 Blender Foundation. All rights reserved.

set(INC
  .
  ../gpu
  ../nodes
  ../imbuf
  ../blenlib
  ../makesdna
  ../makesrna
  ../blenkernel
  ../gpu/intern
  ../../../intern/guardedalloc
)


set(SRC
  intern/compositor_execute.cc
  intern/context.cc
  intern/domain.cc
  intern/result.cc
  intern/scheduler.cc
  intern/texture_pool.cc
  intern/utils.cc

  VPC_compositor_execute.hh
  VPC_context.hh
  VPC_domain.hh
  VPC_result.hh
  VPC_scheduler.hh
  VPC_texture_pool.hh
  VPC_utils.hh
)

set(LIB
  bf_gpu
  bf_nodes
  bf_imbuf
  bf_blenlib
  bf_blenkernel
)

blender_add_lib(bf_viewport_compositor "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")
