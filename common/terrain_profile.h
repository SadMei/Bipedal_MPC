#pragma once

#include <algorithm>
#include <cmath>

struct StairTerrainProfile {
  bool enabled{false};
  double first_riser_x{0.5};
  double tread_depth{0.5};
  double riser_height{0.05};
  double max_height{1.4};

  double stepHeightAt(double x) const {
    if (!enabled || tread_depth <= 1e-9 || riser_height <= 0.0 ||
        x < first_riser_x) {
      return 0.0;
    }
    const double step_index =
        std::floor((x - first_riser_x) / tread_depth) + 1.0;
    return std::clamp(step_index * riser_height, 0.0, max_height);
  }

  int forwardLandingStepIndex(double x, double requested_margin) const {
    if (!enabled || tread_depth <= 1e-9 || riser_height <= 0.0) {
      return 0;
    }
    (void)requested_margin;
    if (x < first_riser_x) {
      return 0;
    }
    const int index = std::max(
        1, static_cast<int>(std::floor((x - first_riser_x) / tread_depth)) +
               1);
    const int max_index = std::max(
        0, static_cast<int>(std::floor(max_height / riser_height + 1e-9)));
    return std::min(index, max_index);
  }

  double safeLandingX(double x, int step_index,
                      double requested_margin) const {
    if (!enabled || tread_depth <= 1e-9) {
      return x;
    }
    const double margin =
        std::clamp(requested_margin, 0.0, 0.45 * tread_depth);
    if (step_index <= 0) {
      return std::min(x, first_riser_x - margin);
    }
    const double lower_edge =
        first_riser_x + (step_index - 1) * tread_depth;
    const double upper_edge = lower_edge + tread_depth;
    return std::clamp(x, lower_edge + margin, upper_edge - margin);
  }

  double stepHeightForIndex(int step_index) const {
    if (!enabled || step_index <= 0) {
      return 0.0;
    }
    return std::clamp(step_index * riser_height, 0.0, max_height);
  }

  double smoothHeightAt(double x) const {
    if (!enabled || tread_depth <= 1e-9 || riser_height <= 0.0) {
      return 0.0;
    }
    const double ramp_start = first_riser_x - tread_depth;
    const double slope = riser_height / tread_depth;
    return std::clamp((x - ramp_start) * slope, 0.0, max_height);
  }
};
