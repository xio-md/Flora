#include "headless_pbr.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <donut/app/Camera.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/math/math.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/Scene.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/View.h>
#include <donut/render/DrawStrategy.h>
#include <donut/render/ForwardShadingPass.h>
#include <donut/render/GeometryPasses.h>
#include <nvrhi/utils.h>

namespace rtxns::python
{
    using donut::app::DeviceManager;
    using donut::engine::CommonRenderPasses;
    using donut::engine::DirectionalLight;
    using donut::engine::FramebufferFactory;
    using donut::engine::PlanarView;
    using donut::engine::Scene;
    using donut::engine::ShaderFactory;
    using donut::engine::TextureCache;
    using donut::render::ForwardShadingPass;
    using donut::vfs::NativeFileSystem;
    using donut::vfs::RootFileSystem;
    using donut::math::radians;

    namespace
    {
        struct DeviceManagerDeleter
        {
            void operator()(DeviceManager* manager) const noexcept
            {
                delete manager;
            }
        };

        dm::float3 to_float3(const std::array<float, 3>& value)
        {
            return dm::float3(value[0], value[1], value[2]);
        }

        dm::float3 normalize_or_throw(dm::float3 value, const char* name)
        {
            const float length_sq = dm::dot(value, value);
            if (length_sq <= 1.0e-12f)
            {
                throw std::runtime_error(std::string(name) + " must be non-zero.");
            }
            return value / std::sqrt(length_sq);
        }

        std::filesystem::path resolve_framework_shader_dir(const std::filesystem::path& runtime_dir)
        {
            if (runtime_dir.empty())
            {
                return {};
            }

            const std::filesystem::path candidates[] = {
                runtime_dir / "bin" / "shaders" / "framework" / "spirv",
                runtime_dir / "shaders" / "framework" / "spirv",
                runtime_dir / "framework" / "spirv"
            };

            for (const auto& candidate : candidates)
            {
                if (std::filesystem::exists(candidate))
                {
                    return candidate;
                }
            }

            return {};
        }
    }

    class RendererContext : public std::enable_shared_from_this<RendererContext>
    {
    public:
        explicit RendererContext(const ContextInitOptions& options)
        {
            if (options.backend != "vulkan")
            {
                throw std::runtime_error("The RTXNS Donut Python backend currently only supports backend='vulkan'.");
            }

            DeviceManager* raw_manager = DeviceManager::Create(nvrhi::GraphicsAPI::VULKAN);
            if (!raw_manager)
            {
                throw std::runtime_error("Failed to create a Vulkan device manager.");
            }

            m_device_manager.reset(raw_manager);

            donut::app::DeviceCreationParameters device_params;
            device_params.adapterIndex = options.device_index;
            device_params.enableDebugRuntime = options.enable_debug;
            device_params.enableNvrhiValidationLayer = options.enable_debug;
            device_params.maxFramesInFlight = 1;
            device_params.swapChainFormat = nvrhi::Format::SRGBA8_UNORM;

            if (!m_device_manager->CreateHeadlessDevice(device_params))
            {
                throw std::runtime_error("Failed to create a headless Vulkan device.");
            }

            m_root_fs = std::make_shared<RootFileSystem>();

            const auto shader_dir = resolve_framework_shader_dir(options.runtime_dir);
            if (!shader_dir.empty())
            {
                m_root_fs->mount("/shaders/donut", shader_dir);
            }

            m_shader_factory = std::make_shared<ShaderFactory>(device(), m_root_fs, "/shaders");
            m_common_passes = std::make_shared<CommonRenderPasses>(device(), m_shader_factory);
        }

        ~RendererContext()
        {
            m_common_passes.reset();
            m_shader_factory.reset();
            m_root_fs.reset();

            if (m_device_manager)
            {
                if (m_device_manager->GetDevice())
                {
                    m_device_manager->GetDevice()->waitForIdle();
                }
                m_device_manager->Shutdown();
            }
        }

        [[nodiscard]] nvrhi::IDevice* device() const
        {
            return m_device_manager->GetDevice();
        }

        [[nodiscard]] const std::shared_ptr<ShaderFactory>& shader_factory() const
        {
            return m_shader_factory;
        }

        [[nodiscard]] const std::shared_ptr<CommonRenderPasses>& common_passes() const
        {
            return m_common_passes;
        }

    private:
        std::unique_ptr<DeviceManager, DeviceManagerDeleter> m_device_manager;
        std::shared_ptr<RootFileSystem> m_root_fs;
        std::shared_ptr<ShaderFactory> m_shader_factory;
        std::shared_ptr<CommonRenderPasses> m_common_passes;
    };

    class HeadlessPbrScene::Impl
    {
    public:
        explicit Impl(std::shared_ptr<RendererContext> context)
            : m_context(std::move(context))
        {
            m_native_fs = std::make_shared<NativeFileSystem>();
            m_texture_cache = std::make_shared<TextureCache>(m_context->device(), m_native_fs, nullptr);
            m_forward_pass = std::make_unique<ForwardShadingPass>(m_context->device(), m_context->common_passes());
            m_forward_pass->Init(*m_context->shader_factory(), ForwardShadingPass::CreateParameters{});

            set_camera(
                {0.0f, 0.5f, 3.0f},
                {0.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                45.0f,
                512,
                512,
                0.1f,
                1000.0f);
        }

        ~Impl()
        {
            if (m_context && m_context->device())
            {
                m_context->device()->waitForIdle();
            }

            m_scene.reset();
            m_texture_cache.reset();
            m_forward_pass.reset();
            m_framebuffer_factory.reset();
            m_readback_target.Reset();
            m_depth_target.Reset();
            m_color_target.Reset();
        }

        void load_scene(const std::filesystem::path& scene_path)
        {
            if (!std::filesystem::exists(scene_path))
            {
                throw std::runtime_error("Scene file does not exist: " + scene_path.string());
            }

            auto* device = m_context->device();
            device->waitForIdle();

            m_scene.reset();
            m_default_light.reset();
            m_texture_cache->Reset();
            device->runGarbageCollection();

            m_scene = std::make_unique<Scene>(
                device,
                *m_context->shader_factory(),
                m_native_fs,
                m_texture_cache,
                nullptr,
                nullptr);

            if (!m_scene->Load(scene_path))
            {
                m_scene.reset();
                throw std::runtime_error("Failed to load scene: " + scene_path.string());
            }

            m_texture_cache->ProcessRenderingThreadCommands(*m_context->common_passes(), 0.0f);
            m_texture_cache->LoadingFinished();

            if (m_default_light_requested || m_scene->GetSceneGraph()->GetLights().empty())
            {
                ensure_default_light_attached();
            }

            m_frame_index = 0;
            m_scene->FinishedLoading(m_frame_index);
        }

        void set_camera(
            const std::array<float, 3>& position,
            const std::array<float, 3>& target,
            const std::array<float, 3>& up,
            float fov_degrees,
            uint32_t width,
            uint32_t height,
            float z_near,
            float z_far)
        {
            if (width == 0 || height == 0)
            {
                throw std::runtime_error("Camera resolution must be positive.");
            }
            if (z_near <= 0.0f || z_far <= z_near)
            {
                throw std::runtime_error("Camera clipping planes are invalid.");
            }
            if (fov_degrees <= 0.0f || fov_degrees >= 179.0f)
            {
                throw std::runtime_error("Camera FOV must be in the range (0, 179).");
            }

            const auto pos = to_float3(position);
            const auto tgt = to_float3(target);
            const auto cam_up = normalize_or_throw(to_float3(up), "up");

            if (dm::length(tgt - pos) <= 1.0e-6f)
            {
                throw std::runtime_error("Camera target must differ from the camera position.");
            }

            resize_targets(width, height);

            m_width = width;
            m_height = height;
            m_z_near = z_near;
            m_z_far = z_far;

            m_camera.LookAt(pos, tgt, cam_up);

            const float aspect = static_cast<float>(width) / static_cast<float>(height);
            m_view.SetViewport(nvrhi::Viewport(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), 0.0f, 1.0f));
            m_view.SetMatrices(
                m_camera.GetWorldToViewMatrix(),
                dm::perspProjD3DStyle(radians(fov_degrees), aspect, z_near, z_far));
            m_view.UpdateCache();
        }

        void set_ambient(
            const std::array<float, 3>& top_rgb,
            const std::array<float, 3>& bottom_rgb)
        {
            m_ambient_top = to_float3(top_rgb);
            m_ambient_bottom = to_float3(bottom_rgb);
        }

        void set_default_light(
            const std::array<float, 3>& direction,
            const std::array<float, 3>& color,
            float irradiance)
        {
            if (irradiance <= 0.0f)
            {
                throw std::runtime_error("Light irradiance must be positive.");
            }

            m_default_light_requested = true;
            m_default_light_direction = normalize_or_throw(to_float3(direction), "direction");
            m_default_light_color = to_float3(color);
            m_default_light_irradiance = irradiance;

            if (m_scene)
            {
                ensure_default_light_attached();
            }
        }

        [[nodiscard]] std::vector<uint8_t> render_frame()
        {
            if (!m_scene)
            {
                throw std::runtime_error("No scene has been loaded.");
            }

            auto* device = m_context->device();
            auto command_list = device->createCommandList();
            command_list->open();

            m_scene->Refresh(command_list, m_frame_index++);

            auto* framebuffer = m_framebuffer_factory->GetFramebuffer(m_view);
            nvrhi::utils::ClearColorAttachment(command_list, framebuffer, 0, nvrhi::Color(0.0f));
            command_list->clearDepthStencilTexture(m_depth_target, nvrhi::AllSubresources, true, 1.0f, false, 0);

            if (m_scene->GetSceneGraph()->GetLights().empty())
            {
                ensure_default_light_attached();
            }

            ForwardShadingPass::Context pass_context;
            const std::vector<std::shared_ptr<donut::engine::LightProbe>> light_probes;
            m_forward_pass->PrepareLights(
                pass_context,
                command_list,
                m_scene->GetSceneGraph()->GetLights(),
                m_ambient_top,
                m_ambient_bottom,
                light_probes);

            donut::render::InstancedOpaqueDrawStrategy opaque_draws;
            donut::render::RenderCompositeView(
                command_list,
                &m_view,
                nullptr,
                *m_framebuffer_factory,
                m_scene->GetSceneGraph()->GetRootNode(),
                opaque_draws,
                *m_forward_pass,
                pass_context,
                "Opaque");

            donut::render::TransparentDrawStrategy transparent_draws;
            donut::render::RenderCompositeView(
                command_list,
                &m_view,
                nullptr,
                *m_framebuffer_factory,
                m_scene->GetSceneGraph()->GetRootNode(),
                transparent_draws,
                *m_forward_pass,
                pass_context,
                "Transparent");

            command_list->setTextureState(m_color_target, nvrhi::AllSubresources, nvrhi::ResourceStates::CopySource);
            command_list->commitBarriers();
            command_list->copyTexture(m_readback_target, nvrhi::TextureSlice(), m_color_target, nvrhi::TextureSlice());
            command_list->setTextureState(m_color_target, nvrhi::AllSubresources, nvrhi::ResourceStates::RenderTarget);
            command_list->commitBarriers();

            command_list->close();
            device->executeCommandList(command_list);
            device->waitForIdle();

            size_t row_pitch = 0;
            const auto* mapped = static_cast<const uint8_t*>(
                device->mapStagingTexture(m_readback_target, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &row_pitch));
            if (!mapped)
            {
                throw std::runtime_error("Failed to map the readback texture.");
            }

            const size_t row_bytes = static_cast<size_t>(m_width) * 4u;
            std::vector<uint8_t> pixels(static_cast<size_t>(m_width) * static_cast<size_t>(m_height) * 4u);

            for (uint32_t row = 0; row < m_height; ++row)
            {
                std::copy_n(
                    mapped + row_pitch * row,
                    row_bytes,
                    pixels.data() + row_bytes * row);
            }

            device->unmapStagingTexture(m_readback_target);
            return pixels;
        }

        [[nodiscard]] uint32_t width() const noexcept
        {
            return m_width;
        }

        [[nodiscard]] uint32_t height() const noexcept
        {
            return m_height;
        }

    private:
        void ensure_default_light_attached()
        {
            if (!m_scene || !m_scene->GetSceneGraph() || !m_scene->GetSceneGraph()->GetRootNode())
            {
                return;
            }

            if (!m_default_light)
            {
                m_default_light = std::make_shared<DirectionalLight>();
                m_scene->GetSceneGraph()->AttachLeafNode(
                    m_scene->GetSceneGraph()->GetRootNode(),
                    m_default_light);
            }

            m_default_light->color = m_default_light_color;
            m_default_light->irradiance = m_default_light_irradiance;
            m_default_light->SetDirection(dm::double3(m_default_light_direction.x, m_default_light_direction.y, m_default_light_direction.z));
        }

        void resize_targets(uint32_t width, uint32_t height)
        {
            if (width == m_width && height == m_height && m_color_target && m_depth_target && m_readback_target)
            {
                return;
            }

            auto* device = m_context->device();
            device->waitForIdle();

            nvrhi::TextureDesc color_desc;
            color_desc.width = width;
            color_desc.height = height;
            color_desc.dimension = nvrhi::TextureDimension::Texture2D;
            color_desc.debugName = "DonutRenderPy/Color";
            color_desc.format = nvrhi::Format::SRGBA8_UNORM;
            color_desc.isRenderTarget = true;
            color_desc.initialState = nvrhi::ResourceStates::RenderTarget;
            color_desc.keepInitialState = true;

            nvrhi::TextureDesc depth_desc;
            depth_desc.width = width;
            depth_desc.height = height;
            depth_desc.dimension = nvrhi::TextureDimension::Texture2D;
            depth_desc.debugName = "DonutRenderPy/Depth";
            depth_desc.format = nvrhi::Format::D32;
            depth_desc.isRenderTarget = true;
            depth_desc.initialState = nvrhi::ResourceStates::DepthWrite;
            depth_desc.keepInitialState = true;

            m_color_target = device->createTexture(color_desc);
            m_depth_target = device->createTexture(depth_desc);
            m_readback_target = device->createStagingTexture(color_desc, nvrhi::CpuAccessMode::Read);

            m_framebuffer_factory = std::make_shared<FramebufferFactory>(device);
            m_framebuffer_factory->RenderTargets = {m_color_target};
            m_framebuffer_factory->DepthTarget = m_depth_target;
        }

        std::shared_ptr<RendererContext> m_context;
        std::shared_ptr<NativeFileSystem> m_native_fs;
        std::shared_ptr<TextureCache> m_texture_cache;
        std::unique_ptr<Scene> m_scene;
        std::unique_ptr<ForwardShadingPass> m_forward_pass;
        std::shared_ptr<FramebufferFactory> m_framebuffer_factory;
        nvrhi::TextureHandle m_color_target;
        nvrhi::TextureHandle m_depth_target;
        nvrhi::StagingTextureHandle m_readback_target;
        PlanarView m_view;
        donut::app::FirstPersonCamera m_camera;
        std::shared_ptr<DirectionalLight> m_default_light;

        dm::float3 m_ambient_top = dm::float3(0.03f, 0.04f, 0.06f);
        dm::float3 m_ambient_bottom = dm::float3(0.01f, 0.01f, 0.01f);

        bool m_default_light_requested = false;
        dm::float3 m_default_light_direction = normalize_or_throw(dm::float3(-0.4f, -1.0f, -0.6f), "default light direction");
        dm::float3 m_default_light_color = dm::float3(1.0f, 1.0f, 1.0f);
        float m_default_light_irradiance = 2.0f;

        uint32_t m_width = 0;
        uint32_t m_height = 0;
        float m_z_near = 0.1f;
        float m_z_far = 1000.0f;
        uint32_t m_frame_index = 0;
    };

    namespace
    {
        std::mutex g_context_mutex;
        std::shared_ptr<RendererContext> g_context;
    }

    HeadlessPbrScene::HeadlessPbrScene(std::shared_ptr<RendererContext> context)
        : m_impl(std::make_unique<Impl>(std::move(context)))
    {
    }

    HeadlessPbrScene::~HeadlessPbrScene() = default;

    void HeadlessPbrScene::load_scene(const std::filesystem::path& scene_path)
    {
        m_impl->load_scene(scene_path);
    }

    void HeadlessPbrScene::set_camera(
        const std::array<float, 3>& position,
        const std::array<float, 3>& target,
        const std::array<float, 3>& up,
        float fov_degrees,
        uint32_t width,
        uint32_t height,
        float z_near,
        float z_far)
    {
        m_impl->set_camera(position, target, up, fov_degrees, width, height, z_near, z_far);
    }

    void HeadlessPbrScene::set_ambient(
        const std::array<float, 3>& top_rgb,
        const std::array<float, 3>& bottom_rgb)
    {
        m_impl->set_ambient(top_rgb, bottom_rgb);
    }

    void HeadlessPbrScene::set_default_light(
        const std::array<float, 3>& direction,
        const std::array<float, 3>& color,
        float irradiance)
    {
        m_impl->set_default_light(direction, color, irradiance);
    }

    std::vector<uint8_t> HeadlessPbrScene::render_frame()
    {
        return m_impl->render_frame();
    }

    uint32_t HeadlessPbrScene::width() const noexcept
    {
        return m_impl->width();
    }

    uint32_t HeadlessPbrScene::height() const noexcept
    {
        return m_impl->height();
    }

    std::shared_ptr<RendererContext> initialize(const ContextInitOptions& options)
    {
        auto context = std::make_shared<RendererContext>(options);
        std::scoped_lock lock(g_context_mutex);
        g_context = context;
        return context;
    }

    void shutdown()
    {
        std::scoped_lock lock(g_context_mutex);
        g_context.reset();
    }

    std::shared_ptr<HeadlessPbrScene> create_scene()
    {
        std::scoped_lock lock(g_context_mutex);
        auto context = g_context;
        if (!context)
        {
            throw std::runtime_error("The RTXNS Donut Python backend is not initialized. Call init(...) first.");
        }

        return std::make_shared<HeadlessPbrScene>(std::move(context));
    }
}
