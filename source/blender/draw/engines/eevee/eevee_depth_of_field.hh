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
 * Depth of field post process effect.
 *
 * There are 2 methods to achieve this effect.
 * - The first uses projection matrix offsetting and sample accumulation to give
 * reference quality depth of field. But this needs many samples to hide the
 * under-sampling.
 * - The second one is a post-processing based one. It follows the
 * implementation described in the presentation "Life of a Bokeh - Siggraph
 * 2018" from Guillaume Abadie. There are some difference with our actual
 * implementation that prioritize quality.
 */

#include "GPU_uniform_buffer.h"

#include "eevee_sampling.hh"
#include "eevee_shader_shared.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Depth of field
 * \{ */

class DepthOfFieldFX {
};

class DepthOfField {
 private:
  Sampling &sampling_;

  DepthOfFieldData data_;
  GPUUniformBuf *ubo_;

  bool enabled_ = false;
  /** Scene settings that are immutable and not inside data_. */
  float user_overblur_;
  float fx_dof_max_coc_;
  /** Circle of Confusion radius for FX DoF passes. Is in view X direction in [0..1] range. */
  float fx_radius_;
  /** Circle of Confusion radius for jittered DoF. Is in view X direction in [0..1] range. */
  float jitter_radius_;
  /** Focus distance in view space. */
  float focus_distance_;
  /** Use Hiqh Quality (expensive) in-focus gather pass. */
  bool do_hq_slight_focus_;

 public:
  DepthOfField(Sampling &sampling) : sampling_(sampling)
  {
    ubo_ = GPU_uniformbuf_create_ex(sizeof(DepthOfFieldData), nullptr, "DepthOfFieldData");
  };

  ~DepthOfField()
  {
    DRW_UBO_FREE_SAFE(ubo_);
  };

  void init(const ::Camera *cam, const Scene *scene)
  {
    if (cam == nullptr || (cam->dof.flag & CAM_DOF_ENABLED) == 0) {
      enabled_ = false;
      return;
    }

    const SceneEEVEE &sce_eevee = scene->eevee;
    do_hq_slight_focus_ = (sce_eevee.flag & SCE_EEVEE_DOF_HQ_SLIGHT_FOCUS) != 0;
    user_overblur_ = sce_eevee.bokeh_overblur / 100.0f;
    fx_dof_max_coc_ = sce_eevee.bokeh_max_size;
    /* TODO(fclem): Make this dependent of the quality of the gather pass. */
    data_.scatter_coc_threshold = 4.0f;
    data_.scatter_color_threshold = sce_eevee.bokeh_threshold;
    data_.scatter_neighbor_max_color = sce_eevee.bokeh_neighbor_max;
    data_.denoise_factor = sce_eevee.bokeh_denoise_fac;
    enabled_ = true;
  }

  void sync(const Object *camera_object_eval, const Scene *scene)
  {
    if (!enabled_) {
      return;
    }
    const ::Camera *cam = reinterpret_cast<const ::Camera *>(camera_object_eval->data);
    const SceneEEVEE &sce_eevee = scene->eevee;

    data_.bokeh_blades = cam->dof.aperture_blades;
    data_.bokeh_rotation = cam->dof.aperture_rotation;
    data_.bokeh_anisotropic_scale[0] = clamp_f(1.0f / cam->dof.aperture_ratio, 0.00001f, 1.0f);
    data_.bokeh_anisotropic_scale[1] = clamp_f(cam->dof.aperture_ratio, 0.00001f, 1.0f);
    copy_v2_v2(data_.bokeh_anisotropic_scale_inv, data_.bokeh_anisotropic_scale);
    invert_v2(data_.bokeh_anisotropic_scale_inv);

    focus_distance_ = BKE_camera_object_dof_distance(camera_object_eval);
    float fstop = max_ff(cam->dof.aperture_fstop, 1e-5f);
    float aperture = 1.0f / (2.0f * fstop);
    if (cam->type == CAM_PERSP) {
      aperture *= cam->lens * 1e-3f;
    }

    if (cam->type == CAM_ORTHO) {
      /* FIXME: Why is this needed? Some kind of implicit unit conversion? */
      aperture *= 0.1f;
      /* NOTE: this 0.5f factor might be caused by jitter_apply(). */
      aperture *= 0.5f;
      /* Really strange behavior from Cycles but replicating. */
      focus_distance_ += cam->clip_start;
    }

    if (cam->dof.aperture_ratio < 1.0) {
      /* If ratio is scaling the bokeh outwards, we scale the aperture so that
       * the gather kernel size will encompass the maximum axis. */
      aperture /= max_ff(cam->dof.aperture_ratio, 1e-5f);
    }

    if ((sce_eevee.flag & SCE_EEVEE_DOF_JITTER) && (sampling_.dof_ring_count_get() > 0) &&
        (cam->type != CAM_PANO)) {
      /* Compute a minimal overblur radius to fill the gaps between the samples.
       * This is just the simplified form of dividing the area of the bokeh by
       * the number of samples. */
      float minimal_overblur = 1.0f / sqrtf(sampling_.dof_sample_count_get());

      fx_radius_ = (minimal_overblur + user_overblur_) * aperture;
      /* Avoid dilating the shape. Over-blur only soften. */
      jitter_radius_ = max_ff(0.0f, aperture - fx_radius_);
    }
    else {
      jitter_radius_ = 0.0f;
      fx_radius_ = aperture;
    }
  }

  void jitter_apply(float winmat[4][4], float viewmat[4][4])
  {
    float radius, theta;
    sampling_.dof_disk_sample_get(&radius, &theta);

    /* Bokeh regular polygon shape parameterization. */
    if (data_.bokeh_blades >= 3.0f) {
      theta = circle_to_polygon_angle(data_.bokeh_blades, theta);
      radius *= circle_to_polygon_radius(data_.bokeh_blades, theta);
    }
    radius *= jitter_radius_;
    theta += data_.bokeh_rotation;

    /* Sample in View Space. */
    float sample[2] = {radius * cosf(theta), radius * sinf(theta)};
    mul_v2_v2(sample, data_.bokeh_anisotropic_scale);
    /* Convert to NDC Space. */
    float jitter[3] = {UNPACK2(sample), -focus_distance_};
    float center[3] = {0.0f, 0.0f, -focus_distance_};
    mul_project_m4_v3(winmat, jitter);
    mul_project_m4_v3(winmat, center);

    const bool is_ortho = (winmat[2][3] != -1.0f);
    if (is_ortho) {
      mul_v2_fl(sample, focus_distance_);
    }

    /* Translate origin. */
    sub_v2_v2(viewmat[3], sample);

    /* Skew winmat Z axis. */
    sub_v2_v2v2(jitter, center, jitter);
    add_v2_v2(winmat[2], jitter);
  }

  /**
   * Getters
   **/
  bool do_jitter(void) const
  {
    return enabled_ && jitter_radius_ > 0.0f;
  }

 private:
};

/** \} */

}  // namespace blender::eevee