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
  KTX_error_code ktxerror = ktxTexture_CreateFromMemory(mem, size, 0, &tex);

  if (ktxerror != KTX_SUCCESS) {
    return NULL;
  }

  ktx_size_t offset;
  ktxerror = ktxTexture_GetImageOffset(tex, 0, 0, 0, &offset);

  if (ktxerror != KTX_SUCCESS) {
    ktxTexture_Destroy(tex);
    return NULL;
  }

  ktx_uint8_t *image = ktxTexture_GetData(tex) + offset;

  ktx_uint32_t xsize = tex->baseWidth;
  ktx_uint32_t ysize = tex->baseHeight;

  ImBuf *ibuf = IMB_allocImBuf(xsize, ysize, 32, (int)IB_rect);

  bool flipx = false, flipy = false;

  for (ktx_uint32_t i = 0; i < xsize + ysize; ++i)
    ibuf->rect[i] = image[i];

  char *pValue;
  uint32_t valueLen;
  ktxerror = ktxHashList_FindValue(&tex->kvDataHead, KTX_ORIENTATION_KEY, &valueLen, (void **)&pValue);
  if (ktxerror != KTX_SUCCESS) {
    char cx, cy;
    if (sscanf(pValue, KTX_ORIENTATION2_FMT, &cx, &cy) == 2) {
      flipx = (cx == 'd');
      flipy = (cy == 'd');
    }
  }

  if (flipx && flipy) {
    int i;
    size_t imbuf_size = ibuf->x * ibuf->y;

    for (i = 0; i < imbuf_size / 2; i++) {
      SWAP(unsigned int, ibuf->rect[i], ibuf->rect[imbuf_size - i - 1]);
    }
  }
  else if (flipy) {
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
  ktxTextureCreateInfo createInfo;
  createInfo.glInternalformat = 0x8058; // GL_RGBA8
  createInfo.baseWidth = ibuf->x;
  createInfo.baseHeight = ibuf->y;
  createInfo.baseDepth = 0;
  createInfo.numDimensions = 2;
  // Note: it is not necessary to provide a full mipmap pyramid.
  createInfo.numLevels = 1;
  createInfo.numLayers = 1;
  createInfo.numFaces = 1;
  createInfo.isArray = KTX_FALSE;
  createInfo.generateMipmaps = KTX_FALSE;
  KTX_error_code result;
  ktxTexture2 *tex;
  result = ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &tex);
  if (KTX_SUCCESS != result) {
    return false;
  }

  ktxTexture *texture = ktxTexture(tex);

  ktx_uint32_t level, layer, faceSlice;
  level = 0;
  layer = 0;
  faceSlice = 0;
  result = ktxTexture_SetImageFromMemory(
      texture, level, layer, faceSlice, (ktx_uint8_t*)ibuf->rect, (size_t)ibuf->x * (size_t)ibuf->y * (size_t) 4);

  if (KTX_SUCCESS != result) {
    ktxTexture_Destroy(texture);
    return false;
  }
  result = ktxTexture_WriteToNamedFile(texture, name);
  ktxTexture_Destroy(texture);

  return KTX_SUCCESS == result;
}
