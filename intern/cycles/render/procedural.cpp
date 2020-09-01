/*
 * Copyright 2011-2018 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "procedural.h"

#include "render/scene.h"

#include "util/util_foreach.h"
#include "util/util_progress.h"

CCL_NAMESPACE_BEGIN

NODE_ABSTRACT_DEFINE(Procedural)
{
  NodeType *type = NodeType::add("procedural_base", NULL);
  return type;
}

Procedural::Procedural(const NodeType *type) : Node(type)
{
}

Procedural::~Procedural()
{
}

bool Procedural::is_procedural() const
{
  return true;
}

ProceduralManager::ProceduralManager()
{
  need_update = true;
}

ProceduralManager::~ProceduralManager()
{
}

void ProceduralManager::update(Scene *scene, Progress &progress)
{
  if (!need_update) {
    return;
  }

  progress.set_status("Updating Procedurals");

  foreach (Procedural *procedural, scene->procedurals) {
    if (progress.get_cancel()) {
      return;
    }

    procedural->create(scene);
  }

  if (progress.get_cancel()) {
    return;
  }

  need_update = false;
}

CCL_NAMESPACE_END
