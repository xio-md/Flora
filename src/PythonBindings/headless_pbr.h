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

        [[nodiscard]] std::vector<uint8_t> render_frame();
        [[nodiscard]] uint32_t width() const noexcept;
        [[nodiscard]] uint32_t height() const noexcept;

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };

    std::shared_ptr<RendererContext> initialize(const ContextInitOptions& options);
    void shutdown();
    std::shared_ptr<HeadlessPbrScene> create_scene();
}
