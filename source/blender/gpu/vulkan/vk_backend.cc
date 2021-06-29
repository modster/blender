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

static std::string api_version_get(VkPhysicalDeviceProperties &properties)
{
  uint32_t major = VK_VERSION_MAJOR(properties.driverVersion);
  uint32_t minor = VK_VERSION_MINOR(properties.driverVersion);
  uint32_t patch = VK_VERSION_PATCH(properties.driverVersion);

  std::stringstream version;
  version << major << "." << minor << "." << patch;
  return version.str();
}

enum class VendorID : uint32_t {
  AMD = 0x1002,
  Intel = 0x1F96,
  NVIDIA = 0x10de,
};
static constexpr StringRef VENDOR_NAME_AMD = "Advanced Micro Devices";
static constexpr StringRef VENDOR_NAME_INTEL = "Intel";
static constexpr StringRef VENDOR_NAME_NVIDIA = "NVIDIA";
static constexpr StringRef VENDOR_NAME_UNKNOWN = "Unknown";

static constexpr StringRef vendor_name_get(VendorID vendor_id)
{
  switch (vendor_id) {
    case VendorID::AMD:
      return VENDOR_NAME_AMD;
    case VendorID::Intel:
      return VENDOR_NAME_INTEL;
    case VendorID::NVIDIA:
      return VENDOR_NAME_NVIDIA;
  }
  return VENDOR_NAME_UNKNOWN;
}

static VendorID vendor_id_get(VkPhysicalDeviceProperties &properties)
{
  return static_cast<VendorID>(properties.vendorID);
}

void VKBackend::platform_init(void)
{
  BLI_assert(!GPG.initialized);

  VKContext *context = VKContext::get();
  VkPhysicalDeviceProperties physical_device_properties;
  VkPhysicalDevice physical_device = context->physical_device_get();
  vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);
  const VendorID vendor_id = vendor_id_get(physical_device_properties);
  const std::string vendor = vendor_name_get(vendor_id);
  const std::string renderer = physical_device_properties.deviceName;
  const std::string version = api_version_get(physical_device_properties).c_str();
  eGPUDeviceType device = GPU_DEVICE_ANY;
  eGPUOSType os = GPU_OS_ANY;
  eGPUDriverType driver = GPU_DRIVER_ANY;
  eGPUSupportLevel support_level = GPU_SUPPORT_LEVEL_SUPPORTED;

#ifdef _WIN32
  os = GPU_OS_WIN;
#elif defined(__APPLE__)
  os = GPU_OS_MAC;
#else
  os = GPU_OS_UNIX;
#endif

  /* TODO(jbakker): extract the driver type. */
  if (vendor_id == VendorID::AMD) {
    device = GPU_DEVICE_ATI;
  }
  else if (vendor_id == VendorID::Intel) {
    device = GPU_DEVICE_INTEL;
  }
  else if (vendor_id == VendorID::NVIDIA) {
    device = GPU_DEVICE_NVIDIA;
  }

  GPG.init(device, os, driver, support_level, vendor.c_str(), renderer.c_str(), version.c_str());
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
