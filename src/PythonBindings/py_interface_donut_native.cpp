#include "py_bindings_common.h"

PYBIND11_MODULE(DonutRenderPyNative, m)
{
    bind_rtxns_headless_pbr_module(m);
}
