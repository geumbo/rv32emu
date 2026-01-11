/*
 * BitNetMCU on rv32emu - Main entry point
 *
 * Runs CNNMNIST inference with BNRV quantization on MNIST test set.
 * Loads images from mnist_test.bin via file I/O.
 * Measures cycle count for performance comparison.
 */

#include <stdint.h>
#include "sim_stdlib.h"

// Model and inference
#include "BitNetMCU_inference.c"
#include "BitNetMCU_model.h"

// Profiler counters
uint64_t cnn_cycles = 0;      // Conv2d layers (L2, L4, L7)
uint64_t pool_cycles = 0;     // MaxPool layers (L6, L9)
uint64_t fc_2bit_cycles = 0;  // FC layer L11 (2bitsym)
uint64_t fc_bnrv_cycles = 0;  // FC layers L13/L15 (BNRV)
uint64_t norm_cycles = 0;     // ReLUNorm

// Forward declarations
uint32_t BitMnistInference(int8_t *input);

// Test configuration
#define NUM_IMAGES 10000
#define IMAGE_SIZE 256
#define RECORD_SIZE (1 + IMAGE_SIZE)  // [label 1B][image 256B]

int main(void)
{
    printf("Running with SIMD Level: %d\n", USE_SIMD);
    printf("\n");

    // Open test data file
    int fd = open("mnist_test.bin", O_RDONLY);
    if (fd < 0) {
        printf("Error: Cannot open mnist_test.bin\n");
        return 1;
    }

    uint64_t total_cycles = 0;
    int correct = 0;
    int8_t image[IMAGE_SIZE];
    uint8_t label;

    for (int i = 0; i < NUM_IMAGES; i++) {
        // Read [label 1B][image 256B]
        read(fd, &label, 1);
        read(fd, image, IMAGE_SIZE);

        uint64_t start = get_cycles();
        uint32_t predicted = BitMnistInference(image);
        total_cycles += get_cycles() - start;

        if (predicted == label) {
            correct++;
        }

        // Progress every 1000 images
        if ((i + 1) % 1000 == 0) {
            printf("Progress: %d/%d\n", i + 1, NUM_IMAGES);
        }
    }

    close(fd);

    printf("\n");
    printf("=== Results ===\n");
    printf("Images tested: %d\n", NUM_IMAGES);
    printf("Correct: %d\n", correct);
    printf("Accuracy: %d.%02d", correct * 100 / NUM_IMAGES,
           (correct * 10000 / NUM_IMAGES) % 100);
    putchar('%');
    putchar('\n');
    // Print Profiler Information
    printf("\nAverage cycles per image: %llu\n", total_cycles / 10000);
    printf("\n=== Performance Profiler ===\n");
    printf("Total Cycles:      %llu\n", total_cycles);
    printf("CNN Cycles:        %llu\n", cnn_cycles);
    printf("Pool Cycles:       %llu\n", pool_cycles);
    printf("FC 2bitsym Cycles: %llu\n", fc_2bit_cycles);
    printf("FC BNRV Cycles:    %llu\n", fc_bnrv_cycles);
    printf("Norm Cycles:       %llu\n", norm_cycles);

    return 0;
}

uint32_t BitMnistInference(int8_t *input)
{
    uint64_t t0;
    int32_t layer_out[MAX_N_ACTIVATIONS];
    int8_t layer_in[MAX_N_ACTIVATIONS * 4];

    // Depthwise separable convolution
    int32_t *tmpbuf = (int32_t *) layer_out;
    int32_t *outputptr = (int32_t *) layer_in;

    for (uint32_t channel = 0; channel < L7_out_channels; channel++) {
        // Copy input to temp buffer
        for (uint32_t i = 0; i < 16 * 16; i++) {
            tmpbuf[i] = input[i];
        }

        // Conv2d layers (L2, L4, L7)
        t0 = get_cycles();
        processconv33ReLU(tmpbuf, L2_weights + 9 * channel, L2_incoming_x, 4,
                          tmpbuf);
        processconv33ReLU(tmpbuf, L4_weights + 9 * channel, L4_incoming_x, 4,
                          tmpbuf);
        cnn_cycles += get_cycles() - t0;

        // MaxPool L6
        t0 = get_cycles();
        processmaxpool22(tmpbuf, L6_incoming_x, tmpbuf);
        pool_cycles += get_cycles() - t0;

        // Conv2d L7
        t0 = get_cycles();
        processconv33ReLU(tmpbuf, L7_weights + 9 * channel, L7_incoming_x, 4,
                          tmpbuf);
        cnn_cycles += get_cycles() - t0;

        // MaxPool L9
        t0 = get_cycles();
        outputptr = processmaxpool22(tmpbuf, L9_incoming_x, outputptr);
        pool_cycles += get_cycles() - t0;
    }

    // Normalize to 8-bit for FC layers
    t0 = get_cycles();
    ReLUNorm((int32_t *) layer_in, layer_in,
             L7_out_channels * L9_outgoing_x * L9_outgoing_y);
    norm_cycles += get_cycles() - t0;

    // FC L11 (2bitsym)
    t0 = get_cycles();
    processfclayer(layer_in, L11_weights, L11_bitperweight,
                   L11_incoming_weights, L11_outgoing_weights, layer_out);
    fc_2bit_cycles += get_cycles() - t0;

    t0 = get_cycles();
    ReLUNorm(layer_out, layer_in, L11_outgoing_weights);
    norm_cycles += get_cycles() - t0;

    // FC L13 (BNRV)
    t0 = get_cycles();
    processfclayer(layer_in, L13_weights, L13_bitperweight,
                   L13_incoming_weights, L13_outgoing_weights, layer_out);
    fc_bnrv_cycles += get_cycles() - t0;

    t0 = get_cycles();
    ReLUNorm(layer_out, layer_in, L13_outgoing_weights);
    norm_cycles += get_cycles() - t0;

    // FC L15 (BNRV)
    t0 = get_cycles();
    processfclayer(layer_in, L15_weights, L15_bitperweight,
                   L15_incoming_weights, L15_outgoing_weights, layer_out);
    fc_bnrv_cycles += get_cycles() - t0;

    t0 = get_cycles();
    uint32_t result = ReLUNorm(layer_out, layer_in, L15_outgoing_weights);
    norm_cycles += get_cycles() - t0;

    return result;
}
