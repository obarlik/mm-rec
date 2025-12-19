#pragma once
#include <string>
#include <cstring>
#include <map>

// Include generated shader headers
#include "shaders/matmul.spv.h"
#include "shaders/matmul_vec4.spv.h"
#include "shaders/matmul_16x16.spv.h"
#include "shaders/matmul_2x2.spv.h"
#include "shaders/matmul_32x32.spv.h"
#include "shaders/matmul_4x4.spv.h"
#include "shaders/matmul_8x8.spv.h"
#include "shaders/matmul_fp16.spv.h"
#include "shaders/matmul_packed.spv.h"
#include "shaders/matmul_prefetch.spv.h"
#include "shaders/matmul_regblock.spv.h"
#include "shaders/matmul_subgroup.spv.h"

namespace mm_rec {

struct ShaderData {
    const unsigned char* data;
    unsigned int length;
};

class EmbeddedShaders {
public:
    static ShaderData get(const std::string& name) {
        // Strip path prefix if present (e.g. "src/shaders/matmul.spv" -> "matmul.spv")
        std::string filename = name;
        size_t lastSlash = filename.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            filename = filename.substr(lastSlash + 1);
        }

        if (filename == "matmul.spv") return {matmul_spv, matmul_spv_len};
        if (filename == "matmul_vec4.spv") return {matmul_vec4_spv, matmul_vec4_spv_len};
        if (filename == "matmul_16x16.spv") return {matmul_16x16_spv, matmul_16x16_spv_len};
        if (filename == "matmul_2x2.spv") return {matmul_2x2_spv, matmul_2x2_spv_len};
        if (filename == "matmul_32x32.spv") return {matmul_32x32_spv, matmul_32x32_spv_len};
        if (filename == "matmul_4x4.spv") return {matmul_4x4_spv, matmul_4x4_spv_len};
        if (filename == "matmul_8x8.spv") return {matmul_8x8_spv, matmul_8x8_spv_len};
        if (filename == "matmul_fp16.spv") return {matmul_fp16_spv, matmul_fp16_spv_len};
        if (filename == "matmul_packed.spv") return {matmul_packed_spv, matmul_packed_spv_len};
        if (filename == "matmul_prefetch.spv") return {matmul_prefetch_spv, matmul_prefetch_spv_len};
        if (filename == "matmul_regblock.spv") return {matmul_regblock_spv, matmul_regblock_spv_len};
        if (filename == "matmul_subgroup.spv") return {matmul_subgroup_spv, matmul_subgroup_spv_len};

        return {nullptr, 0};
    }
};

} // namespace mm_rec
