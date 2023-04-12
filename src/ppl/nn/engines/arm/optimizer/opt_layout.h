// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_LAYOUT_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_LAYOUT_H_

#include <functional>
#include <unordered_map>
#include "ppl/common/retcode.h"
#include "ppl/common/types.h"

namespace ppl { namespace nn { namespace arm {

/* ***********************************************************************************  *
 * Optimize Layout Assumption:                                                          *
 *  1. Graph Input layout: fp32.ndarray                                                 *
 *  2. Input format: major format {ndarray, nbcx} | pass-down format                    *
 *  3. Input type:   major float {fp32, fp16} | index {int64} | armv9 {bf16, i8, u8}    *
 *  4. Output format: major format {ndarray, nbcx} | pass-down format                   *
 *  5. Output type:   major float {fp32, fp16} | index {int64} | armv9 {bf16, i8, u8}   *
 *  6. Reorder/Cast: within {fp32, fp16, bf16, int64, i8, u8}.{ndarray, nbcx}           *
 *  -> Total layout: {fp32, fp16, bf16, int64, i8, u8}.{ndarray, nbcx} (rest ignored)   *
 * ***********************************************************************************  */

/* ***********************************************************************************  *
 * Optimize Layout Strategy:                                                            *
 *  1. BFloat16 fallback                                                                *
 *  2. Input format: major format {ndarray, nbcx} | pass-down format                    *
 *  3. Input type:   major float {fp32, fp16} | index {int64} | armv9 {bf16, i8, u8}    *
 *  4. Output format: major format {ndarray, nbcx} | pass-down format                   *
 *  5. Output type:   major float {fp32, fp16} | index {int64} | armv9 {bf16, i8, u8}   *
 *  6. Reorder/Cast: within {fp32, fp16, bf16, int64, i8, u8}.{ndarray, nbcx}           *
 *  -> Total layout: {fp32, fp16, bf16, int64, i8, u8}.{ndarray, nbcx} (rest ignored)   *
 * ***********************************************************************************  */

template<ppl::common::datatype_t... DTs>
inline bool CheckDTypes(int x) {
    constexpr int dtypes[] = {DTs...};

    bool is_allowed = false;
    for (auto i = 0; i < sizeof...(DTs); i++) {
        is_allowed = is_allowed || (dtypes[i] == x);
    }
    return is_allowed;
}

template<ppl::common::dataformat_t... DFs>
inline bool CheckDFormats(int x) {
    constexpr int dformats[] = {DFs...};

    bool is_allowed = false;
    for (auto i = 0; i < sizeof...(DFs); i++) {
        is_allowed = is_allowed || (dformats[i] == x);
    }
    return is_allowed;
}

inline bool CheckMajorFloat_(ppl::common::datatype_t dtype) {
    return (dtype == ppl::common::DATATYPE_FLOAT32) || (dtype == ppl::common::DATATYPE_FLOAT16);
}

inline ppl::common::dataformat_t GetNbcxFormat_(ppl::common::datatype_t dtype) {
    switch (dtype) {
        case ppl::common::DATATYPE_FLOAT32:
            return ppl::common::DATAFORMAT_N4CX;
        case ppl::common::DATATYPE_FLOAT16:
            return ppl::common::DATAFORMAT_N8CX;
        case ppl::common::DATATYPE_BFLOAT16:
            return ppl::common::DATAFORMAT_N4CX;
        case ppl::common::DATATYPE_UNKNOWN:
            return ppl::common::DATAFORMAT_UNKNOWN;
    }
    return ppl::common::DATAFORMAT_NDARRAY; // (1c)
}

inline ppl::common::dataformat_t GetMajorFormat_(ppl::common::datatype_t dtype, ppl::common::dataformat_t dformat) {
    return (dformat == ppl::common::DATAFORMAT_NDARRAY) ? ppl::common::DATAFORMAT_NDARRAY : GetNbcxFormat_(dtype);
}

#define VAR_IN(v) \
    k_ ## v ## _in
#define VAR_OUT(v) \
    k_ ## v ## _out
#define VAR_LAYOUT_PAIR(idt, idf, odt, odf) \
    k_ ## idt ## idf ## _ ## odt ## odf ## _ 

#define DECLARE_TYPE(t, v) \
    static constexpr uint64_t VAR_IN(t)  = ((uint64_t)v << 48); \
    static constexpr uint64_t VAR_OUT(t) = ((uint64_t)v << 16)

#define DECLARE_FORMAT(t, v) \
    static constexpr uint64_t VAR_IN(t)  = ((uint64_t)v << 32); \
    static constexpr uint64_t VAR_OUT(t) = ((uint64_t)v)

#define DECLARE_LAYOUT_PAIR(dt1, df1, dt2, df2) \
    static constexpr uint64_t VAR_LAYOUT_PAIR(dt1, df1, dt2, df2) = VAR_IN(dt1) | VAR_IN(df1) | VAR_OUT(dt2) | VAR_OUT(df2); \
    static constexpr uint64_t VAR_LAYOUT_PAIR(dt2, df2, dt1, df1) = VAR_IN(dt2) | VAR_IN(df2) | VAR_OUT(dt1) | VAR_OUT(df1)

class OptLayoutManager {
public:
    typedef std::function<ppl::common::RetCode(const void*, const int64_t[4], void*)> layout_convert_func_t;

    static OptLayoutManager* Instance() {
        static OptLayoutManager mgr;
        return &mgr;
    };
    ~OptLayoutManager() {};

    static bool Check(ppl::common::datatype_t, ppl::common::dataformat_t, ppl::common::datatype_t, ppl::common::dataformat_t);
    static ppl::common::RetCode ConvertLayout(const void *src, const int64_t shape[4], 
                                              ppl::common::datatype_t, ppl::common::dataformat_t,
                                              ppl::common::datatype_t, ppl::common::dataformat_t,
                                              void *dst);
    static void Register(uint64_t, layout_convert_func_t);

private:
    static std::unordered_map<uint64_t, layout_convert_func_t> layout_conversion_table_;

private:
    OptLayoutManager() {};
};

}}} // namespace ppl::nn::arm

#endif
