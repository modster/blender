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
 */

#pragma once

#include "BLI_assert.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum GPUShaderBlockType {
  GPU_SHADER_BLOCK_CUSTOM = 0,
  GPU_SHADER_BLOCK_3D_COLOR,
  GPU_NUM_SHADER_BLOCK_TYPES, /* Special value, denotes number of structs. */
} GPUShaderBlockType;

typedef struct GPUShaderBlock3dColor {
  float ModelMatrix[4][4];
  float ModelViewProjectionMatrix[4][4];
  float color[4];
  float WorldClipPlanes[6][4];
  int SrgbTransform;
  int _pad[3];
} GPUShaderBlock3dColor;

BLI_STATIC_ASSERT_ALIGN(GPUShaderBlock3dColor, 16)

#ifdef __cplusplus
}
#endif
