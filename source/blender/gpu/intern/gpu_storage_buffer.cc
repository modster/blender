/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. */

/** \file
 * \ingroup gpu
 */

#include "MEM_guardedalloc.h"
#include <cstring>

#include "BLI_blenlib.h"
#include "BLI_math_base.h"

#include "gpu_backend.hh"
#include "gpu_node_graph.h"

#include "GPU_material.h"
#include "GPU_vertex_buffer.h" /* For GPUUsageType. */

#include "GPU_storage_buffer.h"
#include "gpu_storage_buffer_private.hh"

/* -------------------------------------------------------------------- */
/** \name Creation & Deletion
 * \{ */

namespace blender::gpu {

StorageBuf::StorageBuf(size_t size, const char *name)
{
  /* Make sure that UBO is padded to size of vec4 */
  BLI_assert((size % 16) == 0);

  size_in_bytes_ = size;

  BLI_strncpy(name_, name, sizeof(name_));
}

StorageBuf::~StorageBuf()
{
  MEM_SAFE_FREE(data_);
}

}  // namespace blender::gpu

/** \} */

/* -------------------------------------------------------------------- */
/** \name C-API
 * \{ */

using namespace blender::gpu;

GPUStorageBuf *GPU_storagebuf_create_ex(size_t size,
                                        const void *data,
                                        GPUUsageType usage,
                                        const char *name)
{
  StorageBuf *ubo = GPUBackend::get()->storagebuf_alloc(size, usage, name);
  /* Direct init. */
  if (data != nullptr) {
    ubo->update(data);
  }
  return wrap(ubo);
}

void GPU_storagebuf_free(GPUStorageBuf *ubo)
{
  delete unwrap(ubo);
}

void GPU_storagebuf_update(GPUStorageBuf *ubo, const void *data)
{
  unwrap(ubo)->update(data);
}

void GPU_storagebuf_bind(GPUStorageBuf *ubo, int slot)
{
  unwrap(ubo)->bind(slot);
}

void GPU_storagebuf_unbind(GPUStorageBuf *ubo)
{
  unwrap(ubo)->unbind();
}

void GPU_storagebuf_unbind_all()
{
  /* FIXME */
}

/** \} */
