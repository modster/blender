/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2021-2022 Intel Corporation */

#pragma once

// NOTE(sirgienko) List of PCI device ids, which correspond to allowed Intel GPUs
// public source of device IDs :
// https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/include/pci_ids/iris_pci_ids.h
const static std::set<uint32_t> oneapi_allowed_gpu_devices = {
    // Alchemist dGPU, called DG2 before.
#if 1
    0x4f80,
    0x4f81,
    0x4f82,
    0x4f83,
    0x4f84,
    0x4f87,
    0x4f88,
    0x5690,
    0x5691,
    0x5692,
    0x5693,
    0x5694,
    0x5695,
    0x56a0,
    0x56a1,
    0x56a2,
    0x56a5,
    0x56a6,
    0x56b0,
    0x56b1,
#endif
};
