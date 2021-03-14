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

/** \file
 * \ingroup DNA
 */

#pragma once

#include "DNA_view3d_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct XrActionConfig;

typedef struct XrSessionSettings {
  /** Shading settings, struct shared with 3D-View so settings are the same. */
  struct View3DShading shading;

  char _pad[7];

  char base_pose_type; /* eXRSessionBasePoseType */
  /** Object to take the location and rotation as base position from. */
  Object *base_pose_object;
  float base_pose_location[3];
  float base_pose_angle;

  /** View3D draw flags (V3D_OFSDRAW_NONE, V3D_OFSDRAW_SHOW_ANNOTATION, ...). */
  char draw_flags;
  /** Draw style for controller visualization. */
  char controller_draw_style;
  /** The eye (view) used when projecting 3D to 2D (e.g. when performing GPU select). */
  char selection_eye;

  char _pad2;

  /** Clipping distance. */
  float clip_start, clip_end;

  int flag;

  /** Known action configurations. */
  ListBase actionconfigs;
  /** Default configuration. */
  struct XrActionConfig *defaultconf;
  /** Addon configuration. */
  struct XrActionConfig *addonconf;
  /** User configuration. */
  struct XrActionConfig *userconf;

  /** Objects to constrain to XR headset/controller poses. */
  Object *headset_object;
  Object *controller0_object;
  Object *controller1_object;

  char headset_flag;
  char controller0_flag;
  char controller1_flag;
  char _pad3[5];
} XrSessionSettings;

typedef struct XrActionMapItem {
  struct XrActionMapItem *next, *prev;

  /** Unique name. */
  char idname[64];
  /** Type. */
  char type; /** eXrActionType */
  char _pad[1];

  short flag;

  /** Input threshold for float actions. */
  float threshold;

  /** OpenXR user paths. */
  char user_path0[64];
  char user_path1[64];
  /** OpenXR component paths. */
  char component_path0[192];
  char component_path1[192];

  /** Operator to be called on XR events. */
  char op[64];
  /** Operator properties, assigned to ptr->data and can be written to a file. */
  IDProperty *op_properties;
  /** RNA pointer to access properties. */
  struct PointerRNA *op_properties_ptr;
  char op_flag; /* eXrOpFlag */

  /** Pose action properties. */
  char pose_is_controller;
  char _pad2[2];
  float pose_location[3];
  float pose_rotation[3];

  /** Haptic action properties. */
  float haptic_duration;
  float haptic_frequency;
  float haptic_amplitude;
} XrActionMapItem;

/** #XrActionMapItem.flag */
enum {
  XR_AMI_UPDATE = (1 << 0),
};

typedef struct XrActionMap {
  struct XrActionMap *next, *prev;

  /** Unique name. */
  char idname[64];
  /** OpenXR interaction profile path. */
  char profile[256];

  ListBase items; /* XrActionMapItem */
  short selitem;
  short flag;
  char _pad[4];
} XrActionMap;

/** #XrActionMap.flag */
enum {
  XR_ACTIONMAP_UPDATE = (1 << 0),
};

typedef struct XrActionConfig {
  struct XrActionConfig *next, *prev;

  /** Unique name. */
  char idname[64];

  ListBase actionmaps; /* XrActionMap */
  short actactionmap;
  short selactionmap;
  short flag;
  char _pad[2];
} XrActionConfig;

/** #XrActionConfig.flag */
enum {
  XR_ACTIONCONF_USER = (1 << 0), /* User action config. */
};

#define XR_ACTIONCONF_MAX_NAME 64
#define XR_ACTIONMAP_MAX_NAME 64
#define XR_AMI_MAX_NAME 64

/** XR action type. Enum values match those in GHOST_XrActionType enum for consistency. */
typedef enum eXrActionType {
  XR_BOOLEAN_INPUT = 1,
  XR_FLOAT_INPUT = 2,
  XR_VECTOR2F_INPUT = 3,
  XR_POSE_INPUT = 4,
  XR_VIBRATION_OUTPUT = 100,
} eXrActionType;

typedef enum eXrOpFlag {
  XR_OP_PRESS = 0,
  XR_OP_RELEASE = 1,
  XR_OP_MODAL = 2,
} eXrOpFlag;

typedef enum eXrSessionFlag {
  XR_SESSION_USE_POSITION_TRACKING = (1 << 0),
  XR_SESSION_USE_ABSOLUTE_TRACKING = (1 << 1),
} eXrSessionFlag;

typedef enum eXRSessionBasePoseType {
  XR_BASE_POSE_SCENE_CAMERA = 0,
  XR_BASE_POSE_OBJECT = 1,
  XR_BASE_POSE_CUSTOM = 2,
} eXRSessionBasePoseType;

typedef enum eXrSessionControllerDrawStyle {
  XR_CONTROLLER_DRAW_AXES = 0,
  XR_CONTROLLER_DRAW_RAY = 1,
} eXrSessionControllerDrawStyle;

typedef enum eXrSessionEye {
  XR_EYE_LEFT = 0,
  XR_EYE_RIGHT = 1,
} eXrSessionEye;

typedef enum eXrSessionObjectFlag {
  XR_OBJECT_ENABLE = (1 << 0),
  XR_OBJECT_AUTOKEY = (1 << 1),
} eXrSessionObjectFlag;

#ifdef __cplusplus
}
#endif
