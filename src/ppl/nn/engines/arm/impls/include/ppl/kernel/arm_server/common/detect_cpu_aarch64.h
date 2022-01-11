#ifndef PPL_ARM_SERVER_KERNEL_INCLUDE_DETECT_CPU_AARCH64_H_
#define PPL_ARM_SERVER_KERNEL_INCLUDE_DETECT_CPU_AARCH64_H_

#include <stdint.h>
bool ppl_arm_server_check_taishan_v110();
bool ppl_arm_server_check_neoverse_n1();
bool ppl_arm_server_check_phytium_();

bool ppl_arm_server_check_ext_asimd();
bool ppl_arm_server_check_ext_i8mm();
bool ppl_arm_server_check_fp16();
bool ppl_arm_server_check_bf16();

#endif
