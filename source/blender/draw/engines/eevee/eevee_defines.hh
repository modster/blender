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
 * Copyright 2021, Blender Foundation.
 */

/** \file
 * \ingroup eevee
 *
 * List of defines that are shared with the GPUShaderCreateInfos. We do this to avoid
 * dragging larger headers into the createInfo pipeline which would cause problems.
 */

#pragma once

/* Number of items in a culling batch. Needs to be Power of 2. Must be <= to 65536. */
/* Current limiting factor is the sorting phase which is single pass and only sort within a
 * threadgroup which maximum size is 1024. */
#define CULLING_BATCH_SIZE 1024

/**
 * IMPORTANT: Some data packing are tweaked for these values.
 * Be sure to update them accordingly.
 * SHADOW_TILEMAP_RES max is 32 because of the shared bitmaps used for LOD tagging.
 * It is also limited by the maximum thread group size (1024).
 */
#define SHADOW_TILEMAP_RES 16
#define SHADOW_TILEMAP_LOD 4 /* LOG2(SHADOW_TILEMAP_RES) */
#define SHADOW_TILEMAP_PER_ROW 64
#define SHADOW_PAGE_COPY_GROUP_SIZE 32
#define SHADOW_DEPTH_SCAN_GROUP_SIZE 32
#define SHADOW_AABB_TAG_GROUP_SIZE 64
#define SHADOW_MAX_TILEMAP 4096
#define SHADOW_MAX_PAGE 4096
#define SHADOW_PAGE_PER_ROW 64
