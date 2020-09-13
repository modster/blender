/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * Copyright 2020, Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 */

#include "BKE_global.h"

#include "gpu_capabilities_private.hh"
#include "gpu_platform_private.hh"

#include "vk_backend.hh"

namespace blender::gpu {

/* -------------------------------------------------------------------- */
/** \name Platform
 * \{ */

void VKBackend::platform_init(void)
{
  BLI_assert(!GPG.initialized);
  GPG.initialized = true;

#ifdef _WIN32
  GPG.os = GPU_OS_WIN;
#elif defined(__APPLE__)
  GPG.os = GPU_OS_MAC;
#else
  GPG.os = GPU_OS_UNIX;
#endif

  GPG.device = GPU_DEVICE_ANY;
  GPG.driver = GPU_DRIVER_ANY;

  /* Detect support level */
  GPG.support_level = GPU_SUPPORT_LEVEL_SUPPORTED;

  GPG.create_key(GPG.support_level, "vendor", "renderer", "version");
  GPG.create_gpu_name("vendor", "renderer", "version");
}

void VKBackend::platform_exit(void)
{
  BLI_assert(GPG.initialized);
  GPG.clear();
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Capabilities
 * \{ */

void VKBackend::capabilities_init(void)
{
  /* Common Capabilities. */
  GCaps.max_texture_size = 2048;
  GCaps.max_texture_layers = 64;
  GCaps.max_textures_frag = 16;
  GCaps.max_textures_vert = 16;
  GCaps.max_textures_geom = 16;
  GCaps.max_textures = 46;
  GCaps.mem_stats_support = false;
  GCaps.shader_image_load_store_support = false;
}

/** \} */

}  // namespace blender::gpu
