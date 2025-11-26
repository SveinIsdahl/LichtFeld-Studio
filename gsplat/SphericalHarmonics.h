#pragma once

#include <cstdint>

namespace at {
    class Tensor;
}

namespace gsplat {

    /**
     * @brief Computes the number of spherical harmonic coefficients for a given degree.
     * 
     * For degree d, the number of SH basis functions is (d+1)^2.
     * Note: degree==0 means "disabled SH" - produces a single constant basis coefficient.
     * 
     * @param degree The SH degree (0-4 typically supported)
     * @return Number of SH coefficients: (degree+1)^2
     */
    inline uint32_t num_sh_coeffs(uint32_t degree) {
        return (degree + 1) * (degree + 1);
    }

    void launch_spherical_harmonics_fwd_kernel(
        // inputs
        const uint32_t degrees_to_use,
        const at::Tensor dirs,                // [..., 3]
        const at::Tensor coeffs,              // [..., K, 3]
        const at::optional<at::Tensor> masks, // [...]
        // outputs
        at::Tensor colors // [..., 3]
    );

    void launch_spherical_harmonics_bwd_kernel(
        // inputs
        const uint32_t degrees_to_use,
        const at::Tensor dirs,                // [..., 3]
        const at::Tensor coeffs,              // [..., K, 3]
        const at::optional<at::Tensor> masks, // [...]
        const at::Tensor v_colors,            // [..., 3]
        // outputs
        at::Tensor v_coeffs,            // [..., K, 3]
        at::optional<at::Tensor> v_dirs // [..., 3]
    );

} // namespace gsplat