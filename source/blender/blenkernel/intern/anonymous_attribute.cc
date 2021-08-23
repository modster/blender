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
 */

#include "BKE_anonymous_attribute.hh"

using namespace blender::bke;

struct AnonymousAttributeID {
  mutable std::atomic<int> refcount_weak;
  mutable std::atomic<int> refcount_strong;
  std::string debug_name;
};

AnonymousAttributeID *BKE_anonymous_attribute_id_new_weak(const char *debug_name)
{
  AnonymousAttributeID *anonymous_id = new AnonymousAttributeID();
  anonymous_id->debug_name = debug_name;
  anonymous_id->refcount_weak.store(1);
  return anonymous_id;
}

AnonymousAttributeID *BKE_anonymous_attribute_id_new_strong(const char *debug_name)
{
  AnonymousAttributeID *anonymous_id = new AnonymousAttributeID();
  anonymous_id->debug_name = debug_name;
  anonymous_id->refcount_weak.store(1);
  anonymous_id->refcount_strong.store(1);
  return anonymous_id;
}

bool BKE_anonymous_attribute_id_has_strong_references(const AnonymousAttributeID *anonymous_id)
{
  return anonymous_id->refcount_strong.load() >= 1;
}

void BKE_anonymous_attribute_id_increment_weak(const AnonymousAttributeID *anonymous_id)
{
  anonymous_id->refcount_weak.fetch_add(1);
}

void BKE_anonymous_attribute_id_increment_strong(const AnonymousAttributeID *anonymous_id)
{
  anonymous_id->refcount_weak.fetch_add(1);
  anonymous_id->refcount_strong.fetch_add(1);
}

void BKE_anonymous_attribute_id_decrement_weak(const AnonymousAttributeID *anonymous_id)
{
  const int new_refcount = anonymous_id->refcount_weak.fetch_sub(1) - 1;
  if (new_refcount == 0) {
    delete anonymous_id;
  }
}

void BKE_anonymous_attribute_id_decrement_strong(const AnonymousAttributeID *anonymous_id)
{
  anonymous_id->refcount_strong.fetch_sub(1);
  BKE_anonymous_attribute_id_decrement_weak(anonymous_id);
}

const char *BKE_anonymous_attribute_id_debug_name(const AnonymousAttributeID *anonymous_id)
{
  return anonymous_id->debug_name.c_str();
}
