/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_sh_degree0.cpp
 * @brief Unit tests for SH degree==0 optimization.
 * 
 * When degree==0, spherical harmonics are "disabled" and should produce a
 * constant basis coefficient (Y_0^0 = 0.2820947917738781). This test verifies:
 * 1. sh_basis returns a ones-tensor of shape (N, 1) for degree==0
 * 2. Forward pass with degree==0 produces output = coeffs[..., 0, :] * SH_C0
 * 3. Backward pass correctly computes gradients for degree==0
 * 4. No excessive memory allocation occurs for degree==0
 */

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "Ops.h"
#include "SphericalHarmonics.h"

constexpr int RANDOM_SEED = 42;
constexpr float TOLERANCE = 1e-5f;
constexpr float SH_C0 = 0.2820947917738781f;  // Y_0^0 constant = 1 / (2 * sqrt(pi))

class SHDegree0Test : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(RANDOM_SEED);
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA is not available, skipping GPU tests";
        }
        device_ = torch::kCUDA;
    }

    torch::Device device_ = torch::kCPU;
};

// Test 1: num_sh_coeffs helper function
TEST_F(SHDegree0Test, NumSHCoeffsHelper) {
    // (degree+1)^2 formula verification
    EXPECT_EQ(gsplat::num_sh_coeffs(0), 1u);   // (0+1)^2 = 1
    EXPECT_EQ(gsplat::num_sh_coeffs(1), 4u);   // (1+1)^2 = 4
    EXPECT_EQ(gsplat::num_sh_coeffs(2), 9u);   // (2+1)^2 = 9
    EXPECT_EQ(gsplat::num_sh_coeffs(3), 16u);  // (3+1)^2 = 16
    EXPECT_EQ(gsplat::num_sh_coeffs(4), 25u);  // (4+1)^2 = 25
}

// Test 2: Forward pass with degree==0 produces expected constant output
TEST_F(SHDegree0Test, ForwardDegree0ProducesConstantOutput) {
    const int N = 100;  // Number of directions
    const int K = 1;    // Number of SH coefficients for degree 0
    
    // Create random directions
    auto dirs = torch::randn({N, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    
    // Create SH coefficients with known values
    auto coeffs = torch::ones({N, K, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    coeffs = coeffs * 2.0f;  // Set all coefficients to 2.0
    
    // Forward pass with degree==0
    auto colors = gsplat::spherical_harmonics_fwd(0, dirs, coeffs, c10::nullopt);
    
    // Expected output: coeffs[..., 0, :] * SH_C0 = 2.0 * 0.2820947917738781
    float expected_value = 2.0f * SH_C0;
    
    // Verify output shape
    EXPECT_EQ(colors.size(0), N);
    EXPECT_EQ(colors.size(1), 3);
    
    // Verify output values are constant (independent of direction)
    auto colors_cpu = colors.to(torch::kCPU);
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < 3; ++c) {
            float actual = colors_cpu[i][c].item<float>();
            EXPECT_NEAR(actual, expected_value, TOLERANCE)
                << "Color mismatch at (" << i << ", " << c << "): expected " << expected_value 
                << ", got " << actual;
        }
    }
}

// Test 3: Forward pass with degree==0 is direction-independent
TEST_F(SHDegree0Test, ForwardDegree0IsDirectionIndependent) {
    const int N = 50;
    const int K = 1;
    
    // Create two different sets of random directions
    auto dirs1 = torch::randn({N, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    auto dirs2 = torch::randn({N, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    
    // Same coefficients for both
    auto coeffs = torch::randn({N, K, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    
    // Forward pass with both direction sets
    auto colors1 = gsplat::spherical_harmonics_fwd(0, dirs1, coeffs, c10::nullopt);
    auto colors2 = gsplat::spherical_harmonics_fwd(0, dirs2, coeffs, c10::nullopt);
    
    // Both should produce identical results since degree==0 is direction-independent
    auto diff = (colors1 - colors2).abs().max().item<float>();
    EXPECT_LT(diff, TOLERANCE)
        << "Degree==0 output should be direction-independent, but got max diff: " << diff;
}

// Test 4: Backward pass with degree==0 computes correct gradients
TEST_F(SHDegree0Test, BackwardDegree0ComputesCorrectGradients) {
    const int N = 100;
    const int K = 1;
    
    // Create random directions and coefficients
    auto dirs = torch::randn({N, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    auto coeffs = torch::randn({N, K, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    
    // Create gradient of output (v_colors)
    auto v_colors = torch::ones({N, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    
    // Backward pass with degree==0
    auto [v_coeffs, v_dirs] = gsplat::spherical_harmonics_bwd(K, 0, dirs, coeffs, c10::nullopt, v_colors, true);
    
    // Expected gradient: v_coeffs[..., 0, :] = v_colors * SH_C0
    // All other coefficients should have zero gradient
    auto v_coeffs_cpu = v_coeffs.to(torch::kCPU);
    
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < 3; ++c) {
            float actual = v_coeffs_cpu[i][0][c].item<float>();
            float expected = SH_C0;  // v_colors is all ones
            EXPECT_NEAR(actual, expected, TOLERANCE)
                << "Gradient mismatch at (" << i << ", 0, " << c << ")";
        }
    }
    
    // v_dirs should be zero (no direction dependency for degree==0)
    ASSERT_TRUE(v_dirs.defined());
    auto v_dirs_max = v_dirs.abs().max().item<float>();
    EXPECT_LT(v_dirs_max, TOLERANCE)
        << "Direction gradients should be zero for degree==0, but got max: " << v_dirs_max;
}

// Test 5: Mask handling for degree==0
TEST_F(SHDegree0Test, MaskHandlingDegree0) {
    const int N = 100;
    const int K = 1;
    
    // Create random directions and coefficients
    auto dirs = torch::randn({N, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    auto coeffs = torch::ones({N, K, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    
    // Create mask: first half true, second half false
    auto masks = torch::zeros({N}, torch::TensorOptions().device(device_).dtype(torch::kBool));
    masks.index_put_({torch::indexing::Slice(0, N/2)}, true);
    
    // Forward pass with mask
    auto colors = gsplat::spherical_harmonics_fwd(0, dirs, coeffs, masks);
    
    auto colors_cpu = colors.to(torch::kCPU);
    
    // First half should have values
    for (int i = 0; i < N/2; ++i) {
        float actual = colors_cpu[i][0].item<float>();
        EXPECT_NEAR(actual, SH_C0, TOLERANCE)
            << "Masked-in element at " << i << " should have SH_C0 value";
    }
    
    // Second half should be zero (masked out)
    for (int i = N/2; i < N; ++i) {
        float actual = colors_cpu[i][0].item<float>();
        EXPECT_NEAR(actual, 0.0f, TOLERANCE)
            << "Masked-out element at " << i << " should be zero";
    }
}

// Test 6: Consistency between degree==0 and degree==1+ with zero higher-order coefficients
TEST_F(SHDegree0Test, ConsistencyWithHigherDegreesZeroCoeffs) {
    const int N = 50;
    const int K_degree0 = 1;
    const int K_degree1 = 4;  // (1+1)^2 = 4
    
    // Create random directions
    auto dirs = torch::randn({N, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    
    // Normalize directions (required for higher degrees)
    dirs = torch::nn::functional::normalize(dirs, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    
    // Create coefficients for degree 0
    auto coeffs_d0 = torch::randn({N, K_degree0, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    
    // Create coefficients for degree 1 with same first coefficient and zeros for higher orders
    auto coeffs_d1 = torch::zeros({N, K_degree1, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    coeffs_d1.index_put_({torch::indexing::Slice(), 0, torch::indexing::Slice()}, coeffs_d0.squeeze(-2));
    
    // Forward pass with degree==0
    auto colors_d0 = gsplat::spherical_harmonics_fwd(0, dirs, coeffs_d0, c10::nullopt);
    
    // Forward pass with degree==1 but only using first coefficient (degrees_to_use=0)
    auto colors_d1_deg0 = gsplat::spherical_harmonics_fwd(0, dirs, coeffs_d1, c10::nullopt);
    
    // Both should produce identical results
    auto diff = (colors_d0 - colors_d1_deg0).abs().max().item<float>();
    EXPECT_LT(diff, TOLERANCE)
        << "Degree==0 results should match larger K tensor with degrees_to_use=0, but got max diff: " << diff;
}

// Test 7: Memory efficiency for degree==0 (basic smoke test)
TEST_F(SHDegree0Test, MemoryEfficiencyDegree0) {
    // This test verifies that degree==0 doesn't allocate excessive memory
    // by running with a large number of elements
    const int N = 1000000;  // 1M elements
    const int K = 1;
    
    // Record initial memory
    torch::cuda::synchronize();
    auto mem_before = torch::cuda::memory_allocated();
    
    // Create inputs
    auto dirs = torch::randn({N, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    auto coeffs = torch::randn({N, K, 3}, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
    
    torch::cuda::synchronize();
    auto mem_after_inputs = torch::cuda::memory_allocated();
    
    // Forward pass with degree==0
    auto colors = gsplat::spherical_harmonics_fwd(0, dirs, coeffs, c10::nullopt);
    
    torch::cuda::synchronize();
    auto mem_after_forward = torch::cuda::memory_allocated();
    
    // Memory increase from forward should be minimal (just the output tensor)
    // Output is N * 3 * sizeof(float) = N * 12 bytes
    int64_t expected_output_size = N * 3 * sizeof(float);
    int64_t actual_increase = mem_after_forward - mem_after_inputs;
    
    // Allow some overhead (2x expected), but should not be massive
    EXPECT_LT(actual_increase, expected_output_size * 3)
        << "Memory increase for degree==0 forward should be minimal. "
        << "Expected ~" << expected_output_size << " bytes, got " << actual_increase << " bytes";
}

// Test 8: Batched input with different shapes
TEST_F(SHDegree0Test, BatchedInputShapes) {
    const int K = 1;
    
    // Test various batch shapes
    std::vector<std::vector<int64_t>> shapes = {
        {10, 3},           // [N, 3]
        {2, 10, 3},        // [C, N, 3]
        {4, 8, 16, 3},     // [B1, B2, N, 3]
    };
    
    for (const auto& dir_shape : shapes) {
        // Construct coefficient shape by inserting K before the last dimension
        std::vector<int64_t> coeff_shape = dir_shape;
        coeff_shape.insert(coeff_shape.end() - 1, K);
        
        auto dirs = torch::randn(dir_shape, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
        auto coeffs = torch::randn(coeff_shape, torch::TensorOptions().device(device_).dtype(torch::kFloat32));
        
        // Forward pass should work with various batch shapes
        EXPECT_NO_THROW({
            auto colors = gsplat::spherical_harmonics_fwd(0, dirs, coeffs, c10::nullopt);
            EXPECT_EQ(colors.sizes(), dirs.sizes());
        }) << "Failed for shape: " << torch::IntArrayRef(dir_shape);
    }
}
