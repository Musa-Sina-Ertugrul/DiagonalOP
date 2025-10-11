/**
 * @file definitions.h
 * @brief Common definitions and macros for diagonal operations CUDA kernels
 *
 * This header provides type definitions, macros, and constants used across
 * all diagonal operation CUDA implementations.
 */

#ifndef __DEFINITIONS_H__
#define __DEFINITIONS_H__

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

/**
 * @def THREAD_COUNT
 * @brief Number of threads per CUDA block
 *
 * Default value is 256 threads per block for optimal performance.
 * Can be overridden at compile time if needed.
 */
#ifndef THREAD_COUNT
    #define THREAD_COUNT 256
#endif

/** @brief Cast pointer to nv_bfloat16 pointer */
#define BF_PTR(value) reinterpret_cast<nv_bfloat16*>(value)

/** @brief Cast pointer to nv_bfloat162 pointer (vectorized bfloat16) */
#define BFX2_PTR(value) reinterpret_cast<nv_bfloat162*>(value)

/** @brief Cast pointer to half precision (fp16) pointer */
#define H_PTR(value) reinterpret_cast<half*>(value)

/** @brief Cast pointer to 8-bit integer pointer */
#define I8_PTR(value) reinterpret_cast<int8_t*>(value)

/** @brief Cast pointer to half2 pointer (vectorized fp16) */
#define HX2_PTR(value) reinterpret_cast<half2*>(value)

/** @brief Convert argument to string literal */
#define STRINGIFY(x) #x

/** @brief Concatenate two tokens */
#define CONCAT(a, b) a##b

#endif // __DEFINITIONS_H__
