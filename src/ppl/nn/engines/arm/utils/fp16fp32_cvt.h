#ifndef __ST_HPC_PPL_ARM_FP16FP32_H_
#define __ST_HPC_PPL_ARM_FP16FP32_H_
#ifdef __cplusplus
extern "C" {
#endif //! cplusplus

void CvtFp32ToFp16(int counts, void const *src, void *dst);
void CvtFp16ToFp32(int counts, void const *src, void *dst);

#ifdef __cplusplus
}
#endif //! cplusplus
#endif //! __ST_HPC_PPL_ARM_FP16FP32_H_
