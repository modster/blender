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
 */

#pragma once

#include "DRW_render.h"

#include "BKE_camera.h"

#include "DNA_camera_types.h"
#include "DNA_object_types.h"

enum eEEVEECameraType : int32_t {
  ORTHO = 0,
  PERSP,
  PANO_EQUIRECT,
  PANO_EQUISOLID,
  PANO_EQUIDISTANT,
  PANO_MIRROR,
};

typedef struct EEVEE_Camera {
 public:
  eEEVEECameraType projection;

 public:
  void init(Object *camera_object)
  {
    (void)camera_object;
  }

  void init(const DRWView *drw_view)
  {
    (void)drw_view;
  }
} EEVEE_Camera;