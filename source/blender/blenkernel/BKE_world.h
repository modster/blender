/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2001-2002 NaN Holding BV. All rights reserved. */
#pragma once

/** \file
 * \ingroup bke
 */

#ifdef __cplusplus
extern "C" {
#endif

struct Depsgraph;
struct Main;
struct World;

struct World *BKE_world_add(struct Main *bmain, const char *name);
void BKE_world_eval(struct Depsgraph *depsgraph, struct World *world);

struct World *BKE_world_default(void);

void BKE_world_defaults_free_gpu(void);

/* Module */

void BKE_worlds_init(void);
void BKE_worlds_exit(void);

#ifdef __cplusplus
}
#endif
