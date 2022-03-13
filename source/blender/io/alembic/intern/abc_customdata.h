/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2016 Kévin Dietrich. All rights reserved. */
#pragma once

/** \file
 * \ingroup balembic
 */

#include <Alembic/Abc/All.h>
#include <Alembic/AbcGeom/All.h>

#include <map>

#include "BKE_attribute.h"

#include "BLI_color.hh"
#include "BLI_listbase_wrapper.hh"
#include "BLI_math_vec_types.hh"
#include "BLI_span.hh"

struct CacheAttributeMapping;
struct CustomData;
struct ID;
struct MLoop;
struct MLoopUV;
struct MPoly;
struct MVert;
struct Mesh;
struct MCol;

using Alembic::Abc::ICompoundProperty;
using Alembic::Abc::OCompoundProperty;
namespace blender::io::alembic {

class AttributeSelector;

struct UVSample {
  std::vector<Imath::V2f> uvs;
  std::vector<uint32_t> indices;
};

struct CDStreamConfig {
  MLoop *mloop;
  int totloop;

  MPoly *mpoly;
  int totpoly;

  MVert *mvert;
  int totvert;

  MLoopUV *mloopuv;

  CustomData *loopdata;

  bool pack_uvs;

  /* NOTE: the mesh is mostly used for iterating over loops for loop attributes (UVs, MCol, etc.).
   * It would be nice to remove it, in favor of a more generic way to iterate valid attribute
   * indices.
   */
  Mesh *mesh;
  ID *id;

  float weight;
  float time;
  int timesample_index;
  bool use_vertex_interpolation;
  Alembic::AbcGeom::index_t index;
  Alembic::AbcGeom::index_t ceil_index;

  const char **modifier_error_message;

  DomainInfo domain_info[ATTR_DOMAIN_NUM];

  /* For error reporting when reading vertex colors. */
  std::string iobject_full_name;

  const AttributeSelector *attr_selector;

  /* Alembic needs Blender to keep references to C++ objects (the destructors finalize the writing
   * to ABC). The following fields are all used to keep these references. */

  /* Mapping from UV map name to its ABC property, for the 2nd and subsequent UV maps; the primary
   * UV map is kept alive by the Alembic mesh sample itself. */
  std::map<std::string, Alembic::AbcGeom::OV2fGeomParam> abc_uv_maps;

  /* ORCO coordinates, aka Generated Coordinates. */
  Alembic::AbcGeom::OV3fGeomParam abc_orco;

  /* Mapping from vertex color layer name to its Alembic color data. */
  std::map<std::string, Alembic::AbcGeom::OC4fGeomParam> abc_vertex_colors;

  CDStreamConfig()
      : mloop(NULL),
        totloop(0),
        mpoly(NULL),
        totpoly(0),
        totvert(0),
        pack_uvs(false),
        mesh(NULL),
        weight(0.0f),
        time(0.0f),
        index(0),
        ceil_index(0),
        modifier_error_message(NULL),
        attr_selector(nullptr)
  {
  }
};

/* Get the UVs for the main UV property on a OSchema.
 * Returns the name of the UV layer.
 *
 * For now the active layer is used, maybe needs a better way to choose this. */
const char *get_uv_sample(UVSample &sample, const CDStreamConfig &config, CustomData *data);

void write_custom_data(const OCompoundProperty &prop,
                       CDStreamConfig &config,
                       CustomData *data,
                       int data_type);

/* Need special handling for:
 * - creases (vertex/edge)
 * - velocity
 * - generated coordinate
 * - UVs
 * - vertex colors
 */
class GenericAttributeExporter {
  ID *m_id;
  int64_t cd_mask = CD_MASK_ALL;

 public:
  GenericAttributeExporter(ID *id, int64_t cd_mask_) : m_id(id), cd_mask(cd_mask_)
  {
  }

  void export_attributes();

 protected:
  virtual void export_attribute(blender::Span<bool> span,
                                const std::string &name,
                                AttributeDomain domain) = 0;

  virtual void export_attribute(blender::Span<char> span,
                                const std::string &name,
                                AttributeDomain domain) = 0;

  virtual void export_attribute(blender::Span<int> span,
                                const std::string &name,
                                AttributeDomain domain) = 0;

  virtual void export_attribute(blender::Span<float> span,
                                const std::string &name,
                                AttributeDomain domain) = 0;

  virtual void export_attribute(blender::Span<float2> span,
                                const std::string &name,
                                AttributeDomain domain) = 0;

  virtual void export_attribute(blender::Span<float3> span,
                                const std::string &name,
                                AttributeDomain domain) = 0;

  virtual void export_attribute(blender::Span<ColorGeometry4f> span,
                                const std::string &name,
                                AttributeDomain domain) = 0;

  virtual void export_attribute(blender::Span<MLoopUV> span,
                                const std::string &name,
                                AttributeDomain domain) = 0;

  virtual void export_attribute(blender::Span<MCol> span,
                                const std::string &name,
                                AttributeDomain domain) = 0;

  template<typename BlenderDataType>
  void export_customdata_layer(CustomDataLayer *layer, DomainInfo info, AttributeDomain domain)
  {
    BlenderDataType *data = static_cast<BlenderDataType *>(layer->data);
    int64_t size = static_cast<int64_t>(info.length);
    blender::Span<BlenderDataType> data_span(data, size);
    this->export_attribute(data_span, layer->name, domain);
  }

  void export_generated_coordinates(CustomDataLayer *layer,
                                    DomainInfo info,
                                    AttributeDomain domain);

  void export_attribute_for_domain(DomainInfo info, AttributeDomain domain);
};

GenericAttributeExporter *make_attribute_exporter(ID *id,
                                                  int64_t cd_mask,
                                                  OCompoundProperty &prop);

void set_timesample_index(GenericAttributeExporter *exporter, int timesample_index);

void delete_attribute_exporter(GenericAttributeExporter *exporter);

class AttributeSelector {
  /* Name of the velocity attribute, it is ignored since we deal with separately. */
  std::string velocity_attribute = "";

  int read_flags = 0;

  ListBaseWrapper<const CacheAttributeMapping> mappings;

 public:
  AttributeSelector(ListBase *mappings_) : mappings(mappings_)
  {
  }

  void set_velocity_attribute(const char *name);

  void set_read_flags(int flags);

  const CacheAttributeMapping *get_mapping(const std::string &attr_name) const;

  const std::string &velocity_name() const;

  bool uvs_requested() const;

  bool vertex_colors_requested() const;

  bool original_coordinates_requested() const;

  bool select_attribute(const std::string &attr_name) const;
};

void read_arbitrary_attributes(const CDStreamConfig &config,
                               const ICompoundProperty &schema,
                               const Alembic::AbcGeom::IV2fGeomParam &primary_uvs,
                               const Alembic::Abc::ISampleSelector &sample_sel,
                               float velocity_scale);

bool has_animated_attributes(const ICompoundProperty &arb_geom_params);

}  // namespace blender::io::alembic
