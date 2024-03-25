#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <xtensor-python/pyarray.hpp>

#include "ellipsoidMethods.h"
#include "logSumExpMethods.h"
#include "diffOptHelper.h"

namespace py = pybind11;

PYBIND11_MODULE(diffOptCpp, m) {
    xt::import_numpy();
    m.doc() = "diffOptCpp";

    m.def("getDualVariable", &getDualVariable, "getDualVariable based on xtensor.");
    m.def("getGradientGeneral", &getGradientGeneral, "getGradientGeneral based on xtensor.");
    m.def("getGradientAndHessianGeneral", &getGradientAndHessianGeneral, "getGradientAndHessianGeneral based on xtensor.");

    m.def("rimonMethod", &rimonMethod, "Rimon method based on Eigen.");
    m.def("rimonMethodXtensor", &rimonMethodXtensor, "Rimon method based on xtensor.");
    m.def("ellipsoid_F", &ellipsoid_F, "ellipsoid_value based on xtensor.");
    m.def("ellipsoid_dp", &ellipsoid_dp, "ellipsoid_dp based on xtensor.");
    m.def("ellipsoid_dpdp", &ellipsoid_dpdp, "ellipsoid_dpdp based on xtensor.");
    m.def("ellipsoid_dpdpdp", &ellipsoid_dpdpdp, "ellipsoid_dpdpdp based on xtensor.");
    m.def("ellipsoid_dy", &ellipsoid_dy, "ellipsoid_dy based on xtensor.");
    m.def("ellipsoid_dpdy", &ellipsoid_dpdy, "ellipsoid_dpdy based on xtensor.");
    m.def("ellipsoid_dydy", &ellipsoid_dydy, "ellipsoid_dydy based on xtensor.");
    m.def("ellipsoid_dpdpdy", &ellipsoid_dpdpdy, "ellipsoid_dpdpdy based on xtensor.");
    m.def("ellipsoid_dpdydy", &ellipsoid_dpdydy, "ellipsoid_dpdydy based on xtensor.");
    m.def("RDRT_dq", &RDRT_dq, "RDRT_dq based on xtensor.");
    m.def("RDRT_dqdq", &RDRT_dqdq, "RDRT_dqdq based on xtensor.");
    m.def("getGradientEllipsoids", &getGradientEllipsoids, "getGradientEllipsoid based on xtensor.");
    m.def("getGradientAndHessianEllipsoids", &getGradientAndHessianEllipsoids, "getGradientAndHessianEllipsoid based on xtensor.");

    m.def("getLogSumExpDerivatives", &getLogSumExpDerivatives, "getLogSumExpDerivatives based on xtensor.");
    m.def("getGradientAndHessianEllipsoidAndLogSumExp", &getGradientAndHessianEllipsoidAndLogSumExp, "getGradientAndHessianEllipsoidAndLogSumExp based on xtensor.");
}
