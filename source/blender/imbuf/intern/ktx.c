/*
 * ***** BEGIN GPL LICENSE BLOCK *****
 *
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
 * The Original Code is Copyright (C) 2001-2002 by NaN Holding BV.
 * All rights reserved.
 *
 * The Original Code is: all of this file.
 *
 * Contributor(s): none yet.
 *
 * ***** END GPL LICENSE BLOCK *****
 */

/** \file blender/imbuf/intern/ktx.c
 *  \ingroup imbuf
 */

#include "ktx.h"

#include "IMB_imbuf.h"
#include "IMB_imbuf_types.h"
#include "IMB_filetype.h"

#include "BLI_utildefines.h"
#include "BLI_path_util.h"
#include "BLI_sys_types.h"

#include <string.h>
#include <stdlib.h>

static char KTX_HEAD[] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};


bool check_ktx(const unsigned char *mem, size_t size)
{
  return memcmp(KTX_HEAD, mem, sizeof(KTX_HEAD)) == 0;
}

struct ImBuf *imb_loadktx(const unsigned char *mem, size_t size, int flags, char * UNUSED(colorspace))
{
  ktxTexture *tex;
  KTX_error_code ktx_error = ktxTexture_CreateFromMemory(
      mem, size, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &tex);

  if (ktx_error != KTX_SUCCESS) {
    return NULL;
  }

  ktx_size_t offset;
  ktx_error = ktxTexture_GetImageOffset(tex, 0, 0, 0, &offset);

  if (ktx_error != KTX_SUCCESS) {
    ktxTexture_Destroy(tex);
    return NULL;
  }

  const ktx_uint8_t *image = ktxTexture_GetData(tex) + offset;

  ktx_uint32_t x_size = tex->baseWidth;
  ktx_uint32_t y_size = tex->baseHeight;

  ImBuf *ibuf = IMB_allocImBuf(x_size, y_size, 32, (int)IB_rect);

  bool flip_x = false, flip_y = false;

  size_t num_pixels = (size_t)x_size * (size_t)y_size;
  for (size_t i = 0; i < num_pixels; ++i) {
    ktx_uint8_t *c_out = (ktx_uint8_t *)(ibuf->rect + i);
    const ktx_uint8_t *c_in = image + i * 4;

    for (size_t c = 0; c < 4; ++c) {
      c_out[c] = c_in[c];
    }
  }

  const char *pValue;
  uint32_t valueLen;
  ktx_error = ktxHashList_FindValue(&tex->kvDataHead, KTX_ORIENTATION_KEY, &valueLen, (void **)&pValue);
  if (ktx_error == KTX_SUCCESS) {
    char cx, cy;
    if (sscanf(pValue, KTX_ORIENTATION2_FMT, &cx, &cy) == 2) {
      flip_x = (cx == 'd');
      flip_y = (cy == 'd');
    }
  }

  if (flip_x && flip_y) {
    for (size_t i = 0; i < num_pixels / 2; i++) {
      SWAP(unsigned int, ibuf->rect[i], ibuf->rect[num_pixels - i - 1]);
    }
  }
  else if (flip_y) {
    size_t i, j;
    for (j = 0; j < ibuf->y / 2; j++) {
      for (i = 0; i < ibuf->x; i++) {
        SWAP(unsigned int,
             ibuf->rect[i + j * ibuf->x],
             ibuf->rect[i + (ibuf->y - j - 1) * ibuf->x]);
      }
    }
  }

  ktxTexture_Destroy(tex);

  return ibuf;
}


bool imb_savektx(struct ImBuf *ibuf, const char *name, int UNUSED(flags))
{
  ktxTextureCreateInfo create_info;
  create_info.glInternalformat = 0x8058; // GL_RGBA8
  create_info.baseWidth = ibuf->x;
  create_info.baseHeight = ibuf->y;
  create_info.baseDepth = 1;
  create_info.numDimensions = 2;
  // Note: it is not necessary to provide a full mipmap pyramid.
  create_info.numLevels = 1;
  create_info.numLayers = 1;
  create_info.numFaces = 1;
  create_info.isArray = KTX_FALSE;
  create_info.generateMipmaps = KTX_TRUE;
  KTX_error_code result;
  ktxTexture1 *tex;
  result = ktxTexture1_Create(&create_info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &tex);
  if (KTX_SUCCESS != result) {
    return false;
  }

  ktxTexture *texture = ktxTexture(tex);

  ktx_uint32_t level, layer, face_slice;
  level = 0;
  layer = 0;
  face_slice = 0;
  result = ktxTexture_SetImageFromMemory(
      texture, level, layer, face_slice, (ktx_uint8_t*)ibuf->rect, (size_t)ibuf->x * (size_t)ibuf->y * (size_t) 4);

  if (KTX_SUCCESS != result) {
    ktxTexture_Destroy(texture);
    return false;
  }
  result = ktxTexture_WriteToNamedFile(texture, name);
  ktxTexture_Destroy(texture);

  return KTX_SUCCESS == result;
}
