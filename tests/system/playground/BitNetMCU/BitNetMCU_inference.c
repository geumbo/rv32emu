/*
    BitNetMCU inference functions
    @cpldcpu April 2024

    Performs inference on fully connected layer on a very resource constrained
   MCU. 1,2,4 bit weights are supported.

*/

#include "BitNetMCU_inference.h"
#include <stdint.h>
#include "bnrv.h"

/**
 * @brief Applies a ReLU activation function to an array of integers and
 * normalizes the result to 8-bit integers.
 *
 * @param input Pointer to the input array of 32-bit integers.
 * @param output Pointer to the output array of 8-bit integers.
 * @param n_input The number of elements in the input array.
 * @return The position of maximum value found in the input array before
 * applying the ReLU activation.
 */

uint32_t ReLUNorm(int32_t *input, int8_t *output, uint32_t n_input)
{
    int32_t max_val = -INT32_MAX;
    int32_t max_pos = 255;
    uint32_t scale;
    uint32_t shift;
    int32_t rounding;
    int32_t tmp;

    // Find the maximum value in the input array
    for (uint32_t i = 0; i < n_input; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_pos = i;
        }
    }

    // Normalization
    // Dynamic shift according to max value in the input array
    scale = max_val >>
            7;  // define max range, all bits above 7 will be shifted down
    shift = 0;

    while (scale > 0) {
        shift++;
        scale >>= 1;
    }

    // impact of rounding is almost negligible (+0.03% in eval accuracy)
    // But rounding affects mismatch to python inference engine
    rounding = (1 << (shift)) >> 1;
    // Apply ReLU activation and normalize to 8-bit
    for (uint32_t i = 0; i < n_input; i++) {
        // Apply ReLU activation
        if (input[i] < 0) {
            output[i] = 0;
        } else {
            tmp = (input[i] + rounding) >> shift;
            // clipping needed to catch overflow from rounding
            if (tmp > 127) {
                output[i] = 127;
            } else {
                output[i] = tmp;
            }
        }
    }
    return max_pos;
}

/**
 * @brief Processes a fully connected layer in a neural network.
 *
 * This function processes a fully connected layer in a neural network by
 * performing the dot product of the input activations and weights, and stores
 * the result in the output array.
 *
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param bits_per_weight The number of bits per weight.
 * @param n_input The number of input neurons.
 * @param n_output The number of output neurons.
 * @param output Pointer to the output array where the result of the layer is
 * stored.
 */

void processfclayer(int8_t *activations,
                    const uint32_t *weights,
                    int32_t bits_per_weight,
                    uint32_t n_input,
                    uint32_t n_output,
                    int32_t *output)
{
    const uint32_t *weightidx = weights;

    for (uint32_t i = 0; i < n_output; i++) {
        int8_t *activations_idx = activations;
        int32_t sum = 0;

        if (bits_per_weight == 2) {
            // 2bitsym encoding: 16 weights per 32-bit word
            for (uint32_t k = 0; k < n_input; k += 16) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 16; j++) {
                    int32_t in = *activations_idx++;
                    int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in;
                    sum += tmpsum;
                    if (weightChunk & 0x40000000)
                        sum += tmpsum << 1;
                    weightChunk <<= 2;
                }
            }
        } else if (bits_per_weight == 128) {
            // BNRV: 16 trits packed in 32 bits (2-bit encoding)
            // Encoding: 00=0, 01=+1, 11=-1
            const uint32_t *weightidx32 = (const uint32_t *) weights;
            weightidx32 += i * (n_input / 16);

#if USE_SIMD == 32
            // SIMD-32: process 32 activations at a time
            for (uint32_t k = 0; k < n_input; k += 32) {
                uint32_t w0 = *weightidx32++;
                uint32_t w1 = *weightidx32++;
                sum += __bitnetadd32(activations_idx, w0, w1);
                activations_idx += 32;
            }
#else
            // SIMD 0/4/8/16: process 16 activations at a time
            for (uint32_t k = 0; k < n_input; k += 16) {
                uint32_t w = *weightidx32++;
#if USE_SIMD == 0
                // Scalar fallback
                for (uint32_t j = 0; j < 16; j++) {
                    uint32_t t = w & 0x3;
                    if (t == 0x1)
                        sum += *activations_idx;
                    else if (t == 0x3)
                        sum -= *activations_idx;
                    activations_idx++;
                    w >>= 2;
                }
#elif USE_SIMD == 4
                sum += __bitnetadd4(*(int8x4_t *) (activations_idx + 0),
                                    (w >> 0) & 0xFF);
                sum += __bitnetadd4(*(int8x4_t *) (activations_idx + 4),
                                    (w >> 8) & 0xFF);
                sum += __bitnetadd4(*(int8x4_t *) (activations_idx + 8),
                                    (w >> 16) & 0xFF);
                sum += __bitnetadd4(*(int8x4_t *) (activations_idx + 12),
                                    (w >> 24) & 0xFF);
                activations_idx += 16;
#elif USE_SIMD == 8
                sum += __bitnetadd8(*(int8x4_t *) (activations_idx + 0),
                                    *(int8x4_t *) (activations_idx + 4),
                                    (uint16_t) (w & 0xFFFF));
                sum += __bitnetadd8(*(int8x4_t *) (activations_idx + 8),
                                    *(int8x4_t *) (activations_idx + 12),
                                    (uint16_t) (w >> 16));
                activations_idx += 16;
#elif USE_SIMD == 16
                sum += __bitnetadd16(activations_idx, w);
                activations_idx += 16;
#endif
            }
#endif
        }
        output[i] = sum;
    }
}


/**
 * @brief fused 3x3 conv2d and ReLU activation function
 * convo
 * This function processes a 3x3 convolutional layer in a neural network by
 * performing the dot product of the input activations and weights, and stores
 * the result in the output array. The function also applies a ReLU activation
 * function to the result.
 *
 * To simplify the implementation, some assumptions are made:
 * - The kernel size is always 3x3, and the stride is always 1 and padding is
 * always 0.
 * - Only square arrays (x=y) are supported.
 * - Always the full array is processed, no border handling.
 * - The input activations are stored in a 2D array with dimensions (xy_input,
 * xy_input).
 * - The weights are stored in a 2D array with dimensions (3, 3). The weights
 * are assumed to be 8-bit signed integers.
 * - The output is stored in a 2D array with dimensions (xy_input - 2, xy_input
 * - 2).
 *
 * This function is intended to be used in a loop to process multiple channels
 * in parallel. Convolutions can be performed in place, i.e., the output array
 * can be the same as the input activations array.
 *
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param xy_input The number of input neurons.
 * @param n_shift The number of bits to shift the result of the convolution
 * after summation, typically 8.
 * @param output Pointer to the output array where the result of the layer is
 * stored.
 * @return Pointer to the end of the output array.
 */

int32_t *processconv33ReLU(int32_t *activations,
                           const int8_t *weightsin,
                           uint32_t xy_input,
                           uint32_t n_shift,
                           int32_t *output)
{
    // Create SRAM copy of the weights for speed up
    int8_t weights[9];

    for (uint32_t i = 0; i < 9; i++) {
        weights[i] = weightsin[i];
    }

    for (uint32_t i = 0; i < xy_input - 2; i++) {
        int32_t *row = activations + i * xy_input;
        for (uint32_t j = 0; j < xy_input - 2; j++) {
            int32_t sum = 0;
            int32_t *in = row++;

            // Unrolled convolution loop for 3x3 kernel
            sum += weights[0] * in[0] + weights[1] * in[1] + weights[2] * in[2];
            in += xy_input;
            sum += weights[3] * in[0] + weights[4] * in[1] + weights[5] * in[2];
            in += xy_input;
            sum += weights[6] * in[0] + weights[7] * in[1] + weights[8] * in[2];

            // Apply shift and ReLU
            if (sum < 0) {
                sum = 0;  // ReLU
            } else {
                sum = sum >> n_shift;
            }
            *output++ = (int32_t) sum;
        }
    }

    return output;
}

/**
 * @brief maxpool2d 2x2 function
 *
 * This function performs a 2x2 max pooling operation on a 2D array of input
 * activations. The function divides the input activations into 2x2
 * non-overlapping regions and selects the maximum value in each region.
 *
 * To simplify the implementation, some assumptions are made:
 * - The input activations are stored in a 2D array with dimensions (xy_input,
 * xy_input).
 * - The input activations are assumed to be 8-bit signed integers.
 * - The output is stored in a 2D array with dimensions (xy_input / 2, xy_input
 * / 2).
 * - The stride of the max pooling operation is 2.
 * - Padding is not supported, so the input dimensions must be divisible by 2.
 * - Dilation is not supported.
 * - The output array can be the same as the input activations array. (in place
 * operation)
 *
 * @param activations Pointer to the input activations of the layer.
 * @param xy_input The number of input neurons.
 * @param output Pointer to the output array where the result of the layer is
 * stored.
 * @return Pointer to the end of the output array.
 */

int32_t *processmaxpool22(int32_t *activations,
                          uint32_t xy_input,
                          int32_t *output)
{
    uint32_t xy_output = xy_input / 2;

    // Iterate over the output array dimensions
    for (uint32_t i = 0; i < xy_output; i++) {
        int32_t *row = activations + (2 * i) * xy_input;
        for (uint32_t j = 0; j < xy_output; j++) {
            // Find the maximum value in the corresponding 2x2 patch in the
            // input activations
            int32_t max_val;
            max_val = row[0];
            max_val = max_val > row[xy_input] ? max_val : row[xy_input];
            row++;
            max_val = max_val > row[0] ? max_val : row[0];
            max_val = max_val > row[xy_input] ? max_val : row[xy_input];
            row++;
            // Store the maximum value in the output array
            *output++ = max_val;
        }
    }
    return output;
}
