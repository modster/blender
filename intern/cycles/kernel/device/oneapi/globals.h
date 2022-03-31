/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2021-2022 Intel Corporation */

#pragma once

#include "kernel/integrator/state.h"
#include "kernel/types.h"
#include "kernel/util/profiling.h"

CCL_NAMESPACE_BEGIN

/* NOTE(sirgienko) With SYCL we can't declare __constant__ global variable, which will accessable
   from devece code, like it have done for Cycles CUDA backend. So, the backend will allocate this
   "constant" memory regions and store pointers to them in oneAPI context class
*/

struct IntegratorStateGPU;
struct IntegratorQueueCounter;

typedef struct KernelGlobalsGPU {

#define KERNEL_TEX(type, name) const type *name = nullptr;
#include "kernel/textures.h"
#undef KERNEL_TEX
  IntegratorStateGPU *integrator_state;
  const KernelData *__data;
#ifdef WITH_ONEAPI_SYCL_HOST_ENABLED
  size_t nd_item_local_id_0;
  size_t nd_item_local_range_0;
  size_t nd_item_group_0;
  size_t nd_item_group_range_0;

  size_t nd_item_global_id_0;
  size_t nd_item_global_range_0;
#endif
} KernelGlobalsGPU;

typedef ccl_global KernelGlobalsGPU *ccl_restrict KernelGlobals;

#define kernel_data (*(__data))
#define kernel_integrator_state (*(integrator_state))

/* data lookup defines */

#define kernel_tex_fetch(tex, index) tex[index]
#define kernel_tex_array(tex) tex

CCL_NAMESPACE_END
