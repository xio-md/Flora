#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace rtxns::python
{
    struct ContextInitOptions
    {
        std::filesystem::path runtime_dir;
        std::string backend = "vulkan";
        int device_index = -1;
        bool enable_debug = false;
    };

    class RendererContext;

    class HeadlessPbrScene
    {
    public:
        explicit HeadlessPbrScene(std::shared_ptr<RendererContext> context);
        ~HeadlessPbrScene();

        void load_scene(const std::filesystem::path& scene_path);
        void set_camera(
            const std::array<float, 3>& position,
            const std::array<float, 3>& target,
            const std::array<float, 3>& up,
            float fov_degrees,
            uint32_t width,
            uint32_t height,
            float z_near,
            float z_far);
        void set_ambient(
            const std::array<float, 3>& top_rgb,
            const std::array<float, 3>& bottom_rgb);
        void set_default_light(
            const std::array<float, 3>& direction,
            const std::array<float, 3>& color,
            float irradiance);
        void update_node_transform(
            const std::string& name,
            const std::vector<float>& matrix_values);

        void enable_rt_shadows(bool enable);
        void enable_shadow_blur(bool enable);
        void enable_omm(bool enable);
        void set_shadow_samples(uint32_t n);
        void enable_omm_stress(bool enable);
        void set_omm_config(uint32_t subdiv, uint32_t format);

        bool load_omm_cache(const std::string& path);
        bool save_omm_cache(const std::string& path);

        [[nodiscard]] std::vector<uint8_t> render_frame();
        [[nodiscard]] uint32_t width() const noexcept;
        [[nodiscard]] uint32_t height() const noexcept;

        struct FrameStats
        {
            double total_ms = 0.0;
            double raster_ms = 0.0;       // forward shading (opaque + transparent)
            double blas_build_ms = 0.0;   // BLAS build (first frame only)
            double tlas_build_ms = 0.0;   // TLAS build/update
            double shadow_ray_ms = 0.0;   // ray query compute pass
            double composite_ms = 0.0;    // shadow composite pass
            double readback_ms = 0.0;     // GPU finish + texture readback
            bool   rt_shadows_enabled = false;
            bool   as_built_this_frame = false;
        };

        [[nodiscard]] FrameStats get_last_frame_stats() const;

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };

    std::shared_ptr<RendererContext> initialize(const ContextInitOptions& options);
    void shutdown();
    std::shared_ptr<HeadlessPbrScene> create_scene();
}
