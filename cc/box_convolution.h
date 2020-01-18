#include <ciso646> // && -> and, || -> or etc.

enum class Parameter {xMin, xMax, yMin, yMax};

namespace cpu {

void splitParameters(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac);

void splitParametersUpdateGradInput(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac);

void splitParametersAccGradParameters(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac);

template <bool normalize, bool exact>
void boxConvUpdateOutput(
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & area, at::Tensor & input_integrated, at::Tensor & output);

template <bool normalize, bool exact>
void boxConvUpdateGradInput(
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & area, at::Tensor & grad_output_integrated, at::Tensor & tmpArray);

template <bool exact>
void boxConvAccGradParameters(
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & input_integrated, at::Tensor & tmpArray, Parameter parameter);

void clipParameters(
    at::Tensor & paramMin, at::Tensor & paramMax,
    const double reparametrization, const double minSize, const double maxSize);

at::Tensor computeArea(
    at::Tensor x_min, at::Tensor x_max, at::Tensor y_min, at::Tensor y_max,
    const bool exact, const bool needXDeriv = true, const bool needYDeriv = true);

}

namespace gpu {

void splitParameters(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac);

void splitParametersUpdateGradInput(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac);

void splitParametersAccGradParameters(
    at::Tensor & x_min   , at::Tensor & x_max   , at::Tensor & y_min   , at::Tensor & y_max   ,
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac);

template <bool normalize, bool exact>
void boxConvUpdateOutput(
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & area, at::Tensor & input_integrated, at::Tensor & output);

template <bool normalize, bool exact>
void boxConvUpdateGradInput(
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & area, at::Tensor & grad_output_integrated, at::Tensor & tmpArray);

template <bool exact>
void boxConvAccGradParameters(
    at::Tensor & xMinInt , at::Tensor & xMaxInt , at::Tensor & yMinInt , at::Tensor & yMaxInt ,
    at::Tensor & xMinFrac, at::Tensor & xMaxFrac, at::Tensor & yMinFrac, at::Tensor & yMaxFrac,
    at::Tensor & input_integrated, at::Tensor & tmpArray, Parameter parameter);

void clipParameters(
    at::Tensor & paramMin, at::Tensor & paramMax,
    const double reparametrization, const double minSize, const double maxSize);

at::Tensor computeArea(
    at::Tensor x_min, at::Tensor x_max, at::Tensor y_min, at::Tensor y_max,
    const bool exact, const bool needXDeriv = true, const bool needYDeriv = true);

}