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

#include "NOD_shader.h"

#include "BKE_lib_id.h"
#include "BKE_node.h"
#include "BKE_world.h"

#include "eevee_instance.hh"

namespace blender::eevee {

/* -------------------------------------------------------------------- */
/** \name Default Material
 *
 * \{ */

DefaultWorldNodeTree::DefaultWorldNodeTree()
{
  bNodeTree *ntree = ntreeAddTree(NULL, "World Nodetree", ntreeType_Shader->idname);
  bNode *background = nodeAddStaticNode(NULL, ntree, SH_NODE_BACKGROUND);
  bNode *output = nodeAddStaticNode(NULL, ntree, SH_NODE_OUTPUT_WORLD);
  bNodeSocket *background_out = nodeFindSocket(background, SOCK_OUT, "Background");
  bNodeSocket *output_in = nodeFindSocket(output, SOCK_IN, "Surface");
  nodeAddLink(ntree, background, background_out, output, output_in);
  nodeSetActive(ntree, output);

  color_socket_ =
      (bNodeSocketValueRGBA *)nodeFindSocket(background, SOCK_IN, "Color")->default_value;
  ntree_ = ntree;
}

DefaultWorldNodeTree::~DefaultWorldNodeTree()
{
  ntreeFreeEmbeddedTree(ntree_);
  MEM_SAFE_FREE(ntree_);
}

/* Configure a default nodetree with the given world.  */
bNodeTree *DefaultWorldNodeTree::nodetree_get(::World *wo)
{
  /* WARNING: This function is not threadsafe. Which is not a problem for the moment. */
  copy_v3_fl3(color_socket_->value, wo->horr, wo->horg, wo->horb);
  return ntree_;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name World
 *
 * \{ */

void World::sync()
{
  if (inst_.lookdev.sync_world()) {
    return;
  }

  ::World *bl_world = inst_.scene->world;

  if (bl_world == nullptr) {
    bl_world = BKE_world_default();
  }
  else {
    WorldHandle &wo_handle = inst_.sync.sync_world(bl_world);

    if (wo_handle.recalc != 0) {
      inst_.lightprobes.set_world_dirty();
    }
    wo_handle.reset_recalc_flag();
  }

  /* TODO(fclem) This should be detected to scene level. */
  ::World *orig_world = (::World *)DEG_get_original_id(&bl_world->id);
  if (prev_original_world != orig_world) {
    prev_original_world = orig_world;
    inst_.sampling.reset();
  }

  bNodeTree *ntree = (bl_world->nodetree && bl_world->use_nodes) ?
                         bl_world->nodetree :
                         default_tree.nodetree_get(bl_world);

  GPUMaterial *gpumat = inst_.shaders.world_shader_get(bl_world, ntree, MAT_DOMAIN_SURFACE);
  inst_.shading_passes.background.sync(gpumat);
}

/** \} */

}  // namespace blender::eevee
