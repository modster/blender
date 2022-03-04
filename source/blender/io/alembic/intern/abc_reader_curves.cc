/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2016 Kévin Dietrich. All rights reserved. */

/** \file
 * \ingroup balembic
 */

#include "abc_reader_curves.h"
#include "abc_axis_conversion.h"
#include "abc_customdata.h"
#include "abc_reader_transform.h"
#include "abc_util.h"

#include <cstdio>

#include "MEM_guardedalloc.h"

#include "DNA_curves_types.h"
#include "DNA_object_types.h"

#include "BLI_listbase.h"

#include "BKE_curves.h"
#include "BKE_curves.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.h"
#include "BKE_object.h"
#include "BKE_spline.hh"

using Alembic::Abc::FloatArraySamplePtr;
using Alembic::Abc::Int32ArraySamplePtr;
using Alembic::Abc::P3fArraySamplePtr;
using Alembic::Abc::PropertyHeader;
using Alembic::Abc::UcharArraySamplePtr;

using Alembic::AbcGeom::CurvePeriodicity;
using Alembic::AbcGeom::ICompoundProperty;
using Alembic::AbcGeom::ICurves;
using Alembic::AbcGeom::ICurvesSchema;
using Alembic::AbcGeom::IFloatGeomParam;
using Alembic::AbcGeom::IInt16Property;
using Alembic::AbcGeom::ISampleSelector;
using Alembic::AbcGeom::kWrapExisting;

namespace blender::io::alembic {

AbcCurveReader::AbcCurveReader(const Alembic::Abc::IObject &object, ImportSettings &settings)
    : AbcObjectReader(object, settings)
{
  ICurves abc_curves(object, kWrapExisting);
  m_curves_schema = abc_curves.getSchema();

  get_min_max_time(m_iobject, m_curves_schema, m_min_time, m_max_time);
}

bool AbcCurveReader::valid() const
{
  return m_curves_schema.valid();
}

bool AbcCurveReader::accepts_object_type(
    const Alembic::AbcCoreAbstract::ObjectHeader &alembic_header,
    const Object *const ob,
    const char **err_str) const
{
  if (!Alembic::AbcGeom::ICurves::matches(alembic_header)) {
    *err_str =
        "Object type mismatch, Alembic object path pointed to Curves when importing, but not any "
        "more.";
    return false;
  }

  if (ob->type != OB_CURVES) {
    *err_str = "Object type mismatch, Alembic object path points to Curves.";
    return false;
  }

  return true;
}

void AbcCurveReader::readObjectData(Main *bmain,
                                    const AbcReaderManager & /*manager*/,
                                    const Alembic::Abc::ISampleSelector &sample_sel)
{
  Curves *curves = static_cast<Curves *>(BKE_curves_add(bmain, m_data_name.c_str()));

  m_object = BKE_object_add_only_object(bmain, OB_CURVES, m_object_name.c_str());
  m_object->data = curves;

  read_curves_sample(curves, m_curves_schema, sample_sel);

  if (m_settings->always_add_cache_reader || has_animations(m_curves_schema, m_settings)) {
    addCacheModifier();
  }
}

static short get_curve_resolution(const ICurvesSchema &schema,
                                  const Alembic::Abc::ISampleSelector &sample_sel)
{
  ICompoundProperty user_props = schema.getUserProperties();
  if (user_props) {
    const PropertyHeader *header = user_props.getPropertyHeader(ABC_CURVE_RESOLUTION_U_PROPNAME);
    if (header != nullptr && header->isScalar() && IInt16Property::matches(*header)) {
      IInt16Property resolu(user_props, header->getName());
      return resolu.getValue(sample_sel);
    }
  }

  return 1;
}

static short get_curve_order(Alembic::AbcGeom::CurveType abc_curve_type,
                             const UcharArraySamplePtr orders,
                             size_t curve_index)
{
  switch (abc_curve_type) {
    case Alembic::AbcGeom::kCubic:
      return 4;
    case Alembic::AbcGeom::kVariableOrder:
      if (orders && orders->size() > curve_index) {
        return static_cast<short>((*orders)[curve_index]);
      }
      ATTR_FALLTHROUGH;
    case Alembic::AbcGeom::kLinear:
    default:
      return 2;
  }
}

static int get_curve_overlap(Alembic::AbcGeom::CurvePeriodicity periodicity,
                             const P3fArraySamplePtr positions,
                             int idx,
                             int num_verts,
                             short order)
{
  if (periodicity == Alembic::AbcGeom::kPeriodic) {
    /* Check the number of points which overlap, we don't have
     * overlapping points in Blender, but other software do use them to
     * indicate that a curve is actually cyclic. Usually the number of
     * overlapping points is equal to the order/degree of the curve.
     */

    const int start = idx;
    const int end = idx + num_verts;
    int overlap = 0;

    for (int j = start, k = end - order; j < order; j++, k++) {
      const Imath::V3f &p1 = (*positions)[j];
      const Imath::V3f &p2 = (*positions)[k];

      if (p1 != p2) {
        break;
      }

      overlap++;
    }

    /* TODO: Special case, need to figure out how it coincides with knots. */
    if (overlap == 0 && num_verts > 2 && (*positions)[start] == (*positions)[end - 1]) {
      overlap = 1;
    }

    /* There is no real cycles. */
    return overlap;
  }

  /* kNonPeriodic is always assumed to have no overlap. */
  return 0;
}

static int abc_curves_get_total_point_size(const Int32ArraySamplePtr num_vertices)
{
  if (!num_vertices) {
    return 0;
  }

  int result = 0;
  for (size_t i = 0; i < num_vertices->size(); i++) {
    result += (*num_vertices)[i];
  }
  return result;
}

static void read_curves_sample_ex(Curves *curves,
                                  const ICurvesSchema &schema,
                                  const ICurvesSchema::Sample &smp,
                                  const ISampleSelector sample_sel,
                                  const AttributeSelector *attribute_selector,
                                  const float velocity_scale,
                                  const char **err_str)
{
  bke::CurvesGeometry &geometry = bke::CurvesGeometry::wrap(curves->geometry);
  MutableSpan<int> offsets = geometry.offsets();
  MutableSpan<float3> positions_ = geometry.positions();
  MutableSpan<bool> cyclic = geometry.cyclic();

  float *curves_weights = nullptr;
  int *curves_orders = nullptr;

  int *curves_resolution = static_cast<int *>(CustomData_add_layer_named(&geometry.curve_data,
                                                                         CD_PROP_INT32,
                                                                         CD_DEFAULT,
                                                                         nullptr,
                                                                         geometry.curve_size,
                                                                         "resolution"));

  const int resolution = get_curve_resolution(schema, sample_sel);
  for (int64_t i : geometry.curves_range()) {
    curves_resolution[i] = resolution;
  }

  if (!geometry.radius) {
    geometry.radius = static_cast<float *>(CustomData_add_layer_named(
        &geometry.point_data, CD_PROP_FLOAT, CD_DEFAULT, nullptr, geometry.point_size, "radius"));
  }

  /* Knots are not imported anymore. */
  const Int32ArraySamplePtr num_vertices = smp.getCurvesNumVertices();
  const P3fArraySamplePtr positions = smp.getPositions();
  const FloatArraySamplePtr weights = smp.getPositionWeights();
  const CurvePeriodicity periodicity = smp.getWrap();
  const UcharArraySamplePtr orders = smp.getOrders();

  const IFloatGeomParam widths_param = schema.getWidthsParam();
  FloatArraySamplePtr radiuses;

  if (widths_param.valid()) {
    IFloatGeomParam::Sample wsample = widths_param.getExpandedValue(sample_sel);
    radiuses = wsample.getVals();
  }

  const bool do_radius = (radiuses != nullptr) && (radiuses->size() > 1);
  float radius = (radiuses && radiuses->size() == 1) ? (*radiuses)[0] : 1.0f;

  const bool do_weights = (weights != nullptr) && (weights->size() > 1);
  if (do_weights) {
    curves_weights = static_cast<float *>(CustomData_add_layer_named(&geometry.point_data,
                                                                     CD_PROP_FLOAT,
                                                                     CD_DEFAULT,
                                                                     nullptr,
                                                                     geometry.point_size,
                                                                     "nurbs_weight"));
  }

  const bool do_curves_orders = (orders != nullptr) && (orders->size() > 1);
  if (do_curves_orders) {
    curves_orders = static_cast<int *>(CustomData_add_layer_named(&geometry.curve_data,
                                                                  CD_PROP_INT32,
                                                                  CD_DEFAULT,
                                                                  nullptr,
                                                                  geometry.curve_size,
                                                                  "nurbs_order"));
  }

  int offset = 0;
  size_t position_offset = 0;
  for (size_t i = 0; i < num_vertices->size(); i++) {
    const int num_verts = (*num_vertices)[i];
    offsets[i] = offset;

    const int curve_order = get_curve_order(smp.getType(), orders, i);
    if (do_curves_orders) {
      curves_orders[i] = curve_order;
    }

    /* Check if the curve is cyclic. */
    const int overlap = get_curve_overlap(
        periodicity, positions, position_offset, num_verts, curve_order);

    const bool is_cyclic = overlap != 0;
    cyclic[i] = is_cyclic;

    for (int j = 0; j < num_verts - overlap; j++, position_offset++) {
      const Imath::V3f &pos = (*positions)[position_offset];

      copy_zup_from_yup(positions_[position_offset], pos.getValue());

      if (do_radius) {
        radius = (*radiuses)[position_offset];
      }
      geometry.radius[position_offset] = radius;

      if (do_weights) {
        curves_weights[position_offset] = (*weights)[position_offset];
      }
    }

    offset += num_verts;
    /* Skip duplicate positions due to cyclicity. */
    position_offset += overlap;
  }

  offsets[geometry.curve_size] = offset;

  /* Attributes. */
  CDStreamConfig config;
  config.id = &curves->id;
  config.attr_selector = attribute_selector;
  config.time = sample_sel.getRequestedTime();
  config.modifier_error_message = err_str;
  BKE_id_attribute_get_domains(config.id, config.domain_info);

  read_arbitrary_attributes(config, schema, {}, sample_sel, velocity_scale);
}

void AbcCurveReader::read_curves_sample(Curves *curves,
                                        const ICurvesSchema &schema,
                                        const ISampleSelector &sample_sel)
{
  ICurvesSchema::Sample smp;
  try {
    smp = schema.getValue(sample_sel);
  }
  catch (Alembic::Util::Exception &ex) {
    printf("Alembic: error reading curve sample for '%s/%s' at time %f: %s\n",
           m_iobject.getFullName().c_str(),
           schema.getName().c_str(),
           sample_sel.getRequestedTime(),
           ex.what());
    return;
  }

  const Int32ArraySamplePtr num_vertices = smp.getCurvesNumVertices();
  const P3fArraySamplePtr positions = smp.getPositions();

  const int point_size = abc_curves_get_total_point_size(num_vertices);
  const int curve_size = static_cast<int>(num_vertices->size());

  bke::CurvesGeometry &geometry = bke::CurvesGeometry::wrap(curves->geometry);
  geometry.resize(point_size, curve_size);

  read_curves_sample_ex(curves, m_curves_schema, smp, sample_sel, nullptr, 0.0f, nullptr);
}

void AbcCurveReader::read_geometry(GeometrySet &geometry_set,
                                   const Alembic::Abc::ISampleSelector &sample_sel,
                                   const AttributeSelector *attribute_selector,
                                   int /*read_flag*/,
                                   const float velocity_scale,
                                   const char **err_str)
{
  ICurvesSchema::Sample sample;

  try {
    sample = m_curves_schema.getValue(sample_sel);
  }
  catch (Alembic::Util::Exception &ex) {
    *err_str = "Error reading curve sample; more detail on the console";
    printf("Alembic: error reading curve sample for '%s/%s' at time %f: %s\n",
           m_iobject.getFullName().c_str(),
           m_curves_schema.getName().c_str(),
           sample_sel.getRequestedTime(),
           ex.what());
    return;
  }

  const Int32ArraySamplePtr num_vertices = sample.getCurvesNumVertices();
  const P3fArraySamplePtr positions = sample.getPositions();

  Curves *curves = geometry_set.get_curves_for_write();

  const int point_size = abc_curves_get_total_point_size(num_vertices);
  const int curve_size = static_cast<int>(num_vertices->size());

  bke::CurvesGeometry &geometry = bke::CurvesGeometry::wrap(curves->geometry);
  geometry.resize(point_size, curve_size);

  read_curves_sample_ex(
      curves, m_curves_schema, sample, sample_sel, attribute_selector, velocity_scale, err_str);
}

}  // namespace blender::io::alembic
