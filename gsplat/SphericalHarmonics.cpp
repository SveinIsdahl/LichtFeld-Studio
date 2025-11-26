#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"             // where all the macros are defined
#include "Ops.h"                // a collection of all gsplat operators
#include "SphericalHarmonics.h" // where the launch function is declared

namespace gsplat {

    // Y_0^0 constant = 1 / (2 * sqrt(pi))
    constexpr float SH_C0 = 0.2820947917738781f;

    at::Tensor spherical_harmonics_fwd(
        const uint32_t degrees_to_use,
        const at::Tensor dirs,               // [..., 3]
        const at::Tensor coeffs,             // [..., K, 3]
        const at::optional<at::Tensor> masks // [...]
    ) {
        DEVICE_GUARD(dirs);
        CHECK_INPUT(dirs);
        CHECK_INPUT(coeffs);
        if (masks.has_value()) {
            CHECK_INPUT(masks.value());
        }
        TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
        TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");

        // When degree==0, SH reduces to a constant basis: output = coeff[0] * SH_C0.
        // This short-circuit avoids heavy CUDA kernel overhead and large tensor allocations
        // that would otherwise occur even though degree==0 means "disabled SH".
        if (degrees_to_use == 0) {
            // coeffs shape is [..., K, 3], we only use coeffs[..., 0, :] (the first SH band)
            // Extract first coefficient: [..., 0, :] -> [..., 3]
            auto sh0 = coeffs.select(-2, 0);  // [..., 3]
            at::Tensor colors = sh0 * SH_C0;
            
            // Apply mask if provided
            if (masks.has_value()) {
                auto mask = masks.value().unsqueeze(-1);  // [..., 1]
                colors = colors * mask.to(colors.dtype());
            }
            return colors;  // [..., 3]
        }

        at::Tensor colors = at::empty_like(dirs); // [..., 3]

        launch_spherical_harmonics_fwd_kernel(
            degrees_to_use, dirs, coeffs, masks, colors);
        return colors; // [..., 3]
    }

    std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
        const uint32_t K,
        const uint32_t degrees_to_use,
        const at::Tensor dirs,                // [..., 3]
        const at::Tensor coeffs,              // [..., K, 3]
        const at::optional<at::Tensor> masks, // [...]
        const at::Tensor v_colors,            // [..., 3]
        bool compute_v_dirs) {
        DEVICE_GUARD(dirs);
        CHECK_INPUT(dirs);
        CHECK_INPUT(coeffs);
        CHECK_INPUT(v_colors);
        if (masks.has_value()) {
            CHECK_INPUT(masks.value());
        }
        TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension 3");
        TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
        TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
        const uint32_t N = dirs.numel() / 3;

        // When degree==0, SH reduces to a constant basis: output = coeff[0] * SH_C0.
        // Backward: v_coeffs[..., 0, :] = v_colors * SH_C0, all other coeffs are zero.
        // v_dirs is always zero when degree==0 (no direction dependency).
        if (degrees_to_use == 0) {
            at::Tensor v_coeffs = at::zeros_like(coeffs);  // [..., K, 3]
            
            // v_coeffs[..., 0, :] = v_colors * SH_C0
            auto v_sh0 = v_colors * SH_C0;  // [..., 3]
            
            // Apply mask if provided
            if (masks.has_value()) {
                auto mask = masks.value().unsqueeze(-1);  // [..., 1]
                v_sh0 = v_sh0 * mask.to(v_sh0.dtype());
            }
            
            // Set the gradient for the first coefficient
            v_coeffs.select(-2, 0).copy_(v_sh0);
            
            // v_dirs is always zero for degree==0 (no direction dependency)
            at::Tensor v_dirs;
            if (compute_v_dirs) {
                v_dirs = at::zeros_like(dirs);
            }
            
            return std::make_tuple(v_coeffs, v_dirs);
        }

        at::Tensor v_coeffs = at::zeros_like(coeffs);
        at::Tensor v_dirs;
        if (compute_v_dirs) {
            v_dirs = at::zeros_like(dirs);
        }

        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        uint32_t n_elements = N;
        uint32_t shmem_size = 0;
        launch_spherical_harmonics_bwd_kernel(
            degrees_to_use,
            dirs,
            coeffs,
            masks,
            v_colors,
            v_coeffs,
            v_dirs.defined() ? at::optional<at::Tensor>(v_dirs) : c10::nullopt);
        return std::make_tuple(v_coeffs, v_dirs); // [..., K, 3], [..., 3]
    }

} // namespace gsplat
