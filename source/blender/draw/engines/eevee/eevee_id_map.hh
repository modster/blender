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
 * Structures to identify unique data blocks. The keys are unique so we are able to
 * match ids across frame updates.
 */

#pragma once

#include "BKE_duplilist.h"
#include "BLI_ghash.h"
#include "BLI_map.hh"
#include "DNA_object_types.h"
#include "GPU_material.h"

#include "eevee_engine.h"

namespace blender::eevee {

class Instance;

/* -------------------------------------------------------------------- */
/** \name ObjectKey
 *
 * \{ */

/** Unique key to identify each object in the hashmap. */
struct ObjectKey {
  /** Hash value of the key. */
  uint64_t hash_value;
  /** Original Object or source object for duplis. */
  Object *ob;
  /** Original Parent object for duplis. */
  Object *parent;
  /** Dupli objects recursive unique identifier */
  int id[MAX_DUPLI_RECUR];
  /** If object uses particle system hair. */
  bool use_particle_hair;
#ifdef DEBUG
  char name[64];
#endif
  ObjectKey() : ob(nullptr), parent(nullptr){};

  ObjectKey(Object *ob_, Object *parent_, int id_[MAX_DUPLI_RECUR], bool use_particle_hair_)
      : ob(ob_), parent(parent_), use_particle_hair(use_particle_hair_)
  {
    if (id_) {
      memcpy(id, id_, sizeof(id));
    }
    else {
      memset(id, 0, sizeof(id));
    }
    /* Compute hash on creation so we avoid the cost of it for every sync. */
    hash_value = BLI_ghashutil_ptrhash(ob);
    hash_value = BLI_ghashutil_combine_hash(hash_value, BLI_ghashutil_ptrhash(parent));
    for (int i = 0; i < MAX_DUPLI_RECUR; i++) {
      if (id[i] != 0) {
        hash_value = BLI_ghashutil_combine_hash(hash_value, BLI_ghashutil_inthash(id[i]));
      }
      else {
        break;
      }
    }
#ifdef DEBUG
    STRNCPY(name, ob->id.name);
#endif
  }

  ObjectKey(Object *ob, DupliObject *dupli, Object *parent)
      : ObjectKey(ob, parent, dupli ? dupli->persistent_id : nullptr, false){};

  ObjectKey(Object *ob)
      : ObjectKey(ob, DRW_object_get_dupli(ob), DRW_object_get_dupli_parent(ob)){};

  uint64_t hash(void) const
  {
    return hash_value;
  }

  bool operator<(const ObjectKey &k) const
  {
    if (ob != k.ob) {
      return (ob < k.ob);
    }
    if (parent != k.parent) {
      return (parent < k.parent);
    }
    if (use_particle_hair != k.use_particle_hair) {
      return (use_particle_hair < k.use_particle_hair);
    }
    return memcmp(id, k.id, sizeof(id)) < 0;
  }

  bool operator==(const ObjectKey &k) const
  {
    if (ob != k.ob) {
      return false;
    }
    if (parent != k.parent) {
      return false;
    }
    if (use_particle_hair != k.use_particle_hair) {
      return false;
    }
    return memcmp(id, k.id, sizeof(id)) == 0;
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Draw Data
 *
 * \{ */

struct ObjectHandle : public DrawData {
  ObjectKey object_key;

  void reset_recalc_flag(void)
  {
    if (recalc != 0) {
      recalc = 0;
    }
  }
};

struct WorldHandle : public DrawData {
  void reset_recalc_flag(void)
  {
    if (recalc != 0) {
      recalc = 0;
    }
  }
};

class SyncModule {
 private:
  Instance &inst_;

 public:
  SyncModule(Instance &inst) : inst_(inst){};
  ~SyncModule(){};

  ObjectHandle &sync_object(Object *ob);
  WorldHandle &sync_world(::World *world);
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name MaterialKey
 *
 * \{ */

enum eMaterialPipeline {
  MAT_PIPE_DEFERRED = 0,
  MAT_PIPE_FORWARD = 1,
  MAT_PIPE_DEFERRED_PREPASS = 2,
  MAT_PIPE_FORWARD_PREPASS = 3,
  MAT_PIPE_VOLUME = 4,
  MAT_PIPE_SHADOW = 5,
};

enum eMaterialGeometry {
  MAT_GEOM_MESH = 0,
  MAT_GEOM_HAIR = 1,
  MAT_GEOM_GPENCIL = 2,
  MAT_GEOM_VOLUME = 3,
  MAT_GEOM_WORLD = 4,
  MAT_GEOM_LOOKDEV = 5,
};

static inline void material_type_from_shader_uuid(uint64_t shader_uuid,
                                                  eMaterialPipeline &pipeline_type,
                                                  eMaterialGeometry &geometry_type)
{
  const uint64_t geometry_mask = ((1u << 3u) - 1u);
  const uint64_t pipeline_mask = ((1u << 3u) - 1u);
  geometry_type = static_cast<eMaterialGeometry>(shader_uuid & geometry_mask);
  pipeline_type = static_cast<eMaterialPipeline>((shader_uuid >> 3u) & pipeline_mask);
}

static inline uint64_t shader_uuid_from_material_type(eMaterialPipeline pipeline_type,
                                                      eMaterialGeometry geometry_type)
{
  return geometry_type | (pipeline_type << 3);
}

static inline eMaterialGeometry to_material_geometry(const Object *ob)
{
  switch (ob->type) {
    case OB_HAIR:
      return MAT_GEOM_HAIR;
    case OB_VOLUME:
      return MAT_GEOM_VOLUME;
    case OB_GPENCIL:
      return MAT_GEOM_GPENCIL;
    default:
      return MAT_GEOM_MESH;
  }
}

/** Unique key to identify each material in the hashmap. */
struct MaterialKey {
  Material *mat;
  uint64_t options;

  MaterialKey(::Material *mat_, eMaterialGeometry geometry, eMaterialPipeline surface_pipeline)
      : mat(mat_)
  {
    options = shader_uuid_from_material_type(surface_pipeline, geometry);
  }

  uint64_t hash(void) const
  {
    BLI_assert(options < sizeof(*mat));
    return (uint64_t)mat + options;
  }

  bool operator<(const MaterialKey &k) const
  {
    return (mat < k.mat) || (options < k.options);
  }

  bool operator==(const MaterialKey &k) const
  {
    return (mat == k.mat) && (options == k.options);
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name ShaderKey
 *
 * \{ */

struct ShaderKey {
  GPUShader *shader;
  uint64_t options;

  ShaderKey(GPUMaterial *gpumat, eMaterialGeometry geometry, eMaterialPipeline pipeline)
  {
    shader = GPU_material_get_shader(gpumat);
    options = shader_uuid_from_material_type(pipeline, geometry);
  }

  uint64_t hash(void) const
  {
    return (uint64_t)shader + options;
  }

  bool operator<(const ShaderKey &k) const
  {
    return (shader < k.shader) || (options < k.options);
  }

  bool operator==(const ShaderKey &k) const
  {
    return (shader == k.shader) && (options == k.options);
  }
};

/** \} */

}  // namespace blender::eevee
