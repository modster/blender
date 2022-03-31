/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#pragma once

#include "util/string.h"
#include "util/vector.h"

#include "kernel/device/oneapi/kernel.h"

#ifdef WITH_ONEAPI
struct oneAPIDLLInterface {
#  define DLL_INTERFACE_CALL(function, return_type, ...) \
    return_type (*function)(__VA_ARGS__) = nullptr;
#  include "kernel/device/oneapi/dll_interface_template.h"
#  undef DLL_INTERFACE_CALL
};
#endif

CCL_NAMESPACE_BEGIN

class Device;
class DeviceInfo;
class Profiler;
class Stats;

bool device_oneapi_init();

Device *device_oneapi_create(const DeviceInfo &info, Stats &stats, Profiler &profiler);

void device_oneapi_info(vector<DeviceInfo> &devices);

string device_oneapi_capabilities();

CCL_NAMESPACE_END
