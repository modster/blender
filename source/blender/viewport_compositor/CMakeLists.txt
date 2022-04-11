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
  intern/compiler.cc
  intern/context.cc
  intern/conversion_processor_operation.cc
  intern/domain.cc
  intern/evaluator.cc
  intern/gpu_material_node.cc
  intern/gpu_material_operation.cc
  intern/node_operation.cc
  intern/operation.cc
  intern/processor_operation.cc
  intern/realize_on_domain_processor_operation.cc
  intern/reduce_to_single_value_processor_operation.cc
  intern/result.cc
  intern/scheduler.cc
  intern/texture_pool.cc
  intern/utilities.cc

  VPC_compiler.hh
  VPC_context.hh
  VPC_conversion_processor_operation.hh
  VPC_domain.hh
  VPC_evaluator.hh
  VPC_gpu_material_node.hh
  VPC_gpu_material_operation.hh
  VPC_input_descriptor.hh
  VPC_node_operation.hh
  VPC_operation.hh
  VPC_processor_operation.hh
  VPC_realize_on_domain_processor_operation.hh
  VPC_reduce_to_single_value_processor_operation.hh
  VPC_result.hh
  VPC_scheduler.hh
  VPC_texture_pool.hh
  VPC_utilities.hh
)

set(LIB
  bf_gpu
  bf_nodes
  bf_imbuf
  bf_blenlib
  bf_blenkernel
)

blender_add_lib(bf_viewport_compositor "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")
