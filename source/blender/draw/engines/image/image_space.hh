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
 * \ingroup draw_engine
 */

#pragma once

class ShaderParameters;

/**
 *  Space accessor.
 *
 *  Image engine is used to draw the images inside multiple spaces \see SpaceLink.
 *  The AbstractSpaceAccessor is an interface to communicate with a space.
 */
class AbstractSpaceAccessor {
 public:
  virtual ~AbstractSpaceAccessor() = default;

  /**
   * Return the active image of the space.
   *
   * The returned image will be drawn in the space.
   *
   * The return value is optional.
   */
  virtual Image *get_image(Main *bmain) = 0;

  /**
   * Return the #ImageUser of the space.
   *
   * The return value is optional.
   */
  virtual ImageUser *get_image_user() = 0;

  /**
   * Acquire the image buffer of the image.
   *
   * \param image: Image to get the buffer from. Image is the same as returned from the #get_image
   * member.
   * \param lock: pointer to a lock object.
   * \return Image buffer of the given image.
   */
  virtual ImBuf *acquire_image_buffer(Image *image, void **lock) = 0;

  /**
   * Release a previous locked image from #acquire_image_buffer.
   */
  virtual void release_buffer(Image *image, ImBuf *image_buffer, void *lock) = 0;

  /**
   * Update the r_shader_parameters with space specific settings.
   *
   * Only update the #ShaderParameters.flags and #ShaderParameters.shuffle. Other parameters
   * are updated inside the image engine.
   */
  virtual void get_shader_parameters(ShaderParameters &r_shader_parameters,
                                     ImBuf *image_buffer,
                                     bool is_tiled) = 0;

  /**
   * Retrieve the gpu textures to draw.
   */
  virtual void get_gpu_textures(Image *image,
                                ImageUser *iuser,
                                ImBuf *image_buffer,
                                GPUTexture **r_gpu_texture,
                                bool *r_owns_texture,
                                GPUTexture **r_tex_tile_data) = 0;

  /**
   * Override the view for drawing.
   */
  virtual DRWView *create_view_override(const ARegion *UNUSED(region)) = 0;

  /**
   * Initialize the matrix that will be used to draw the image. The matrix will be send as object
   * matrix to the drawing pipeline.
   */
  virtual void get_image_mat(const ImBuf *image_buffer,
                             const ARegion *region,
                             float r_mat[4][4]) const = 0;

  /** \brief Is (wrap) repeat option enabled in the space. */
  virtual bool use_tile_drawing() const = 0;
};  // namespace blender::draw::image_engine
