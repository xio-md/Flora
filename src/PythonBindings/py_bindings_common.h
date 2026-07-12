#pragma once

#include "headless_pbr.h"

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

inline void bind_rtxns_headless_pbr_module(py::module_ &m)
{
    m.doc() = "Headless Vulkan PBR renderer bindings for the RTXNS Donut backend.";

    py::class_<rtxns::python::HeadlessPbrScene, std::shared_ptr<rtxns::python::HeadlessPbrScene>>(m, "Scene")
        .def("load_scene",
            [](rtxns::python::HeadlessPbrScene &self, const std::string &path)
            {
                py::gil_scoped_release release;
                self.load_scene(path);
            },
            py::arg("path"))
        .def("set_camera",
            &rtxns::python::HeadlessPbrScene::set_camera,
            py::arg("position"),
            py::arg("target"),
            py::arg("up"),
            py::arg("fov_degrees"),
            py::arg("width"),
            py::arg("height"),
            py::arg("z_near") = 0.1f,
            py::arg("z_far") = 1000.0f)
        .def("set_ambient",
            &rtxns::python::HeadlessPbrScene::set_ambient,
            py::arg("top_rgb"),
            py::arg("bottom_rgb"))
        .def("set_default_light",
            &rtxns::python::HeadlessPbrScene::set_default_light,
            py::arg("direction"),
            py::arg("color") = std::array<float, 3>{1.0f, 1.0f, 1.0f},
            py::arg("irradiance") = 2.0f)
        .def("update_node_transform",
            &rtxns::python::HeadlessPbrScene::update_node_transform,
            py::arg("name"),
            py::arg("matrix_values"))
        .def("enable_rt_shadows",
            &rtxns::python::HeadlessPbrScene::enable_rt_shadows,
            py::arg("enable"))
        .def("enable_shadow_blur",
            &rtxns::python::HeadlessPbrScene::enable_shadow_blur,
            py::arg("enable"))
        .def("enable_omm",
            &rtxns::python::HeadlessPbrScene::enable_omm,
            py::arg("enable"))
        .def("set_shadow_samples",
            &rtxns::python::HeadlessPbrScene::set_shadow_samples,
            py::arg("n"))
        .def("enable_omm_stress",
            &rtxns::python::HeadlessPbrScene::enable_omm_stress,
            py::arg("enable"))
        .def("set_omm_config",
            &rtxns::python::HeadlessPbrScene::set_omm_config,
            py::arg("subdiv"), py::arg("format"))
        .def("load_omm_cache",
            [](rtxns::python::HeadlessPbrScene &self, const std::string& path)
            {
                py::gil_scoped_release release;
                return self.load_omm_cache(path);
            },
            py::arg("path"))
        .def("save_omm_cache",
            [](rtxns::python::HeadlessPbrScene &self, const std::string& path)
            {
                py::gil_scoped_release release;
                return self.save_omm_cache(path);
            },
            py::arg("path"))
        .def("add_camera",
            &rtxns::python::HeadlessPbrScene::add_camera,
            py::arg("position"),
            py::arg("target"),
            py::arg("up"),
            py::arg("fov_degrees"),
            py::arg("width"),
            py::arg("height"),
            py::arg("z_near") = 0.1f,
            py::arg("z_far") = 1000.0f)
        .def("set_camera_at",
            &rtxns::python::HeadlessPbrScene::set_camera_at,
            py::arg("index"),
            py::arg("position"),
            py::arg("target"),
            py::arg("up"),
            py::arg("fov_degrees"),
            py::arg("width"),
            py::arg("height"),
            py::arg("z_near") = 0.1f,
            py::arg("z_far") = 1000.0f)
        .def_property_readonly("camera_count", &rtxns::python::HeadlessPbrScene::camera_count)
        .def("render_frame",
            [](rtxns::python::HeadlessPbrScene &self, int camera_index)
            {
                auto pixels = [&]()
                {
                    py::gil_scoped_release release;
                    return self.render_frame(static_cast<uint32_t>(camera_index));
                }();

                return py::bytes(
                    reinterpret_cast<const char *>(pixels.data()),
                    static_cast<py::ssize_t>(pixels.size()));
            },
            py::arg("camera_index") = 0)
        .def("render_frame_batch",
            [](rtxns::python::HeadlessPbrScene &self, const std::vector<uint32_t>& indices)
            {
                auto frames = [&]()
                {
                    py::gil_scoped_release release;
                    return self.render_frame_batch(indices);
                }();

                py::list out;
                for (const auto& pixels : frames)
                    out.append(py::bytes(
                        reinterpret_cast<const char*>(pixels.data()),
                        static_cast<py::ssize_t>(pixels.size())));
                return out;
            },
            py::arg("camera_indices"))
        .def("submit_frame_batch",
            &rtxns::python::HeadlessPbrScene::submit_frame_batch,
            py::arg("camera_indices"))
        .def("submit_frame_batch_ex",
            &rtxns::python::HeadlessPbrScene::submit_frame_batch_ex,
            py::arg("camera_indices"),
            py::arg("micro_batch_size"))
        .def("is_batch_ready",
            &rtxns::python::HeadlessPbrScene::is_batch_ready,
            py::arg("token"))
        .def("read_frame_batch",
            [](rtxns::python::HeadlessPbrScene &self, uint64_t token)
            {
                auto frames = [&]()
                {
                    py::gil_scoped_release release;
                    return self.read_frame_batch(token);
                }();

                py::list out;
                for (const auto& pixels : frames)
                    out.append(py::bytes(
                        reinterpret_cast<const char*>(pixels.data()),
                        static_cast<py::ssize_t>(pixels.size())));
                return out;
            },
            py::arg("token"))
        .def("set_readback_ring_depth",
            &rtxns::python::HeadlessPbrScene::set_readback_ring_depth,
            py::arg("depth"))
        .def_property_readonly("readback_ring_depth",
            &rtxns::python::HeadlessPbrScene::get_readback_ring_depth)
        .def("get_last_frame_stats",
            [](const rtxns::python::HeadlessPbrScene &self)
            {
                const auto& s = self.get_last_frame_stats();
                py::dict d;
                d["total_ms"] = s.total_ms;
                d["raster_ms"] = s.raster_ms;
                d["blas_build_ms"] = s.blas_build_ms;
                d["tlas_build_ms"] = s.tlas_build_ms;
                d["shadow_ray_ms"] = s.shadow_ray_ms;
                d["composite_ms"] = s.composite_ms;
                d["readback_ms"] = s.readback_ms;
                d["rt_shadows_enabled"] = s.rt_shadows_enabled;
                d["as_built_this_frame"] = s.as_built_this_frame;
                return d;
            })
        .def_property_readonly("width", &rtxns::python::HeadlessPbrScene::width)
        .def_property_readonly("height", &rtxns::python::HeadlessPbrScene::height);

    m.def("init",
        [](const std::string &runtime_dir, const std::string &backend, int device_index, bool enable_debug)
        {
            rtxns::python::ContextInitOptions options;
            options.runtime_dir = runtime_dir;
            options.backend = backend;
            options.device_index = device_index;
            options.enable_debug = enable_debug;

            py::gil_scoped_release release;
            rtxns::python::initialize(options);
        },
        py::arg("runtime_dir") = "",
        py::arg("backend") = "vulkan",
        py::arg("device_index") = -1,
        py::arg("enable_debug") = false);

    m.def("create_scene", &rtxns::python::create_scene);

    m.def("destroy",
        []()
        {
            py::gil_scoped_release release;
            rtxns::python::shutdown();
        });
}
