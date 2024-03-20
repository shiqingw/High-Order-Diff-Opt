#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <xtensor-python/pyarray.hpp>

#include "ellipsoidMethods.h"
#include "diffOptHelper.h"

namespace py = pybind11;

PYBIND11_MODULE(diffOptEllipsoidCpp, m) {
    xt::import_numpy();
    m.doc() = "diffOptEllipsoidCpp";
    m.def("rimonMethod", &rimonMethod, "Rimon method based on Eigen.");
    m.def("rimonMethodXtensor", &rimonMethodXtensor, "Rimon method based on xtensor.");
    m.def("F", &F, "F based on xtensor.");
    m.def("F_dp", &F_dp, "F_dp based on xtensor.");
    m.def("F_dpdp", &F_dpdp, "F_dpdp based on xtensor.");
    m.def("F_dpdpdp", &F_dpdpdp, "F_dpdpdp based on xtensor.");
    m.def("F_dy", &F_dy, "F_dy based on xtensor.");
    m.def("F_dpdy", &F_dpdy, "F_dpdy based on xtensor.");
    m.def("F_dydy", &F_dydy, "F_dydy based on xtensor.");
    m.def("F_dpdpdy", &F_dpdpdy, "F_dpdpdy based on xtensor.");
    m.def("F_dpdydy", &F_dpdydy, "F_dpdydy based on xtensor.");
    m.def("RDRT_dq", &RDRT_dq, "F_dpdydy based on xtensor.");
    m.def("RDRT_dqdq", &RDRT_dqdq, "F_dpdydy based on xtensor.");
    m.def("getDualVariable", &getDualVariable, "getDualVariable based on xtensor.");
    m.def("getGradientGeneral", &getGradientGeneral, "getGradientGeneral based on xtensor.");
    m.def("getGradientAndHessianGeneral", &getGradientAndHessianGeneral, "getGradientAndHessianGeneral based on xtensor.");
    m.def("getGradientEllipsoid", &getGradientEllipsoid, "getGradientEllipsoid based on xtensor.");

}
