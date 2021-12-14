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
 * The Original Code is Copyright (C) 2021 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup gpu
 *
 * Descriptior type used to define shader structure, resources and interfaces.
 */

#include "BLI_map.hh"
#include "BLI_set.hh"
#include "BLI_string_ref.hh"

#include "GPU_texture.h"

#include "gpu_shader_create_info.hh"
#include "gpu_shader_create_info_private.hh"
#undef GPU_SHADER_INTERFACE_INFO
#undef GPU_SHADER_CREATE_INFO

namespace blender::gpu::shader {

using CreateInfoDictionnary = Map<StringRef, ShaderCreateInfo *>;
using InterfaceDictionnary = Map<StringRef, StageInterfaceInfo *>;

static CreateInfoDictionnary *g_create_infos = nullptr;
static InterfaceDictionnary *g_interfaces = nullptr;

}  // namespace blender::gpu::shader

using namespace blender::gpu::shader;

void gpu_shader_create_info_init()
{
  g_create_infos = new CreateInfoDictionnary();
  g_interfaces = new InterfaceDictionnary();

#define GPU_SHADER_INTERFACE_INFO(_interface, _inst_name) \
  auto *ptr_##_interface = new StageInterfaceInfo(#_interface, _inst_name); \
  auto &_interface = *ptr_##_interface; \
  g_interfaces->add_new(#_interface, ptr_##_interface); \
  _interface

#define GPU_SHADER_CREATE_INFO(_descriptor) \
  auto *ptr_##_descriptor = new ShaderCreateInfo(#_descriptor); \
  auto &_descriptor = *ptr_##_descriptor; \
  g_create_infos->add_new(#_descriptor, ptr_##_descriptor); \
  _descriptor

/* Declare, register and construct the infos. */
#include "gpu_shader_create_info_list.hh"

/* Baked shader data appended to create infos. */
#ifdef GPU_RUNTIME
#  include "gpu_shader_baked.hh"
#endif
}

void gpu_shader_create_info_exit()
{
  for (auto value : g_create_infos->values()) {
    delete value;
  }
  delete g_create_infos;

  for (auto value : g_interfaces->values()) {
    delete value;
  }
  delete g_interfaces;
}

/* Runtime descriptors are not registered in the descriptor dictionnary and cannot be searched. */
const GPUShaderCreateInfo *gpu_shader_create_info_get(const char *info_name)
{
  blender::gpu::shader::ShaderCreateInfo *info = g_create_infos->lookup(info_name);
  return reinterpret_cast<const GPUShaderCreateInfo *>(info);
}
