/*
This is part of OpenLoong Dynamics Control, an open project for the control of biped robot,
Copyright (C) 2024 Humanoid Robot (Shanghai) Co., Ltd, under Apache 2.0.
Feel free to use in any purpose, and cite OpenLoong-Dynamics-Control in any style, to contribute to the advancement of the community.
 <https://atomgit.com/openloong/openloong-dyn-control.git>
 <web@openloong.org.cn>
*/
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include "GLFW_callbacks.h"
#include "MJ_interface.h"
#include "PVT_ctrl.h"
#include "data_logger.h"
#include "data_bus.h"
#include "pino_kin_dyn.h"
#include "useful_math.h"
#include "wbc_priority.h"
#include "mpc.h"
#include "gait_scheduler.h"
#include "foot_placement.h"
#include "joystick_interpreter.h"
#include <cstdint>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

const double dt = 0.001;
const double dt_200Hz = 0.005;
char error[1000] = "Could not load binary model";
const std::string scene_model_path = [] {
  const char *configured_path = std::getenv("ODC_SCENE_XML");
  return configured_path != nullptr ? std::string(configured_path)
                                    : std::string("../models/scene.xml");
}();
mjModel *mj_model = mj_loadXML(scene_model_path.c_str(), 0, error, 1000);
mjData *mj_data = mj_model != nullptr ? mj_makeData(mj_model) : nullptr;

namespace {
double computeVelTrackError(const DataBus &robot_state) {
  return (robot_state.dq.block<2, 1>(0, 0) -
          robot_state.js_vel_des.block<2, 1>(0, 0))
      .norm();
}

double computeTorsoAngleError(const DataBus &robot_state) {
  return std::sqrt(robot_state.base_rpy(0) * robot_state.base_rpy(0) +
                   robot_state.base_rpy(1) * robot_state.base_rpy(1));
}

bool detectFall(const DataBus &robot_state, double min_height,
                double max_torso_angle) {
  return robot_state.q(2) < min_height ||
         std::abs(robot_state.base_rpy(0)) > max_torso_angle ||
         std::abs(robot_state.base_rpy(1)) > max_torso_angle;
}

const char *legStateName(DataBus::LegState state) {
  switch (state) {
  case DataBus::LSt:
    return "LSt";
  case DataBus::RSt:
    return "RSt";
  case DataBus::DSt:
    return "DSt";
  }
  return "Unknown";
}

struct OneStepPredictionFrame {
  bool valid{false};
  double time{0.0};
  double phi{0.0};
  DataBus::LegState leg_state{DataBus::DSt};
  Eigen::Vector3d rpy{Eigen::Vector3d::Zero()};
  Eigen::Vector3d omega{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d nominal_inertia{Eigen::Matrix3d::Identity()};
  Eigen::Matrix3d inertia{Eigen::Matrix3d::Identity()};
  Eigen::Matrix3d inertia_dot_raw{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d inertia_dot_filtered{Eigen::Matrix3d::Zero()};
  Eigen::Vector3d external_force_impulse{Eigen::Vector3d::Zero()};
  Eigen::Vector3d external_moment_impulse{Eigen::Vector3d::Zero()};
  double integrated_dt{0.0};
  int contact_count_sum{0};
  int physics_samples{0};
};

struct PendingMpcPrediction {
  double origin_time{0.0};
  double target_time{0.0};
  int horizon_steps{0};
  double origin_phi{0.0};
  DataBus::LegState origin_leg_state{DataBus::DSt};
  double origin_wz_ref{0.0};
  Eigen::Vector3d start_omega{Eigen::Vector3d::Zero()};
  Eigen::Vector3d predicted_omega{Eigen::Vector3d::Zero()};
};

struct CompletedMpcPrediction {
  PendingMpcPrediction prediction;
  double actual_time{0.0};
  Eigen::Vector3d actual_omega{Eigen::Vector3d::Zero()};
};

OneStepPredictionFrame makePredictionFrame(const DataBus &state,
                                           double sim_time,
                                           const Eigen::Matrix3d &nominal_inertia,
                                           const Eigen::Matrix3d &filtered_inertia_dot,
                                           const Eigen::Vector3d &true_omega_W) {
  OneStepPredictionFrame frame;
  frame.valid = true;
  frame.time = sim_time;
  frame.phi = state.phi;
  frame.leg_state = state.legState;
  frame.rpy = state.base_rpy;
  frame.omega = true_omega_W;
  frame.nominal_inertia = nominal_inertia;
  frame.inertia = state.inertia;
  frame.inertia_dot_raw = state.inertia_dot;
  frame.inertia_dot_filtered = filtered_inertia_dot;
  return frame;
}

enum class OmegaPredictionModel { SRBM, VariableInertia, InertiaRate, InertiaRateRaw };

Eigen::Vector3d predictOmegaOneStep(const OneStepPredictionFrame &frame,
                                    OmegaPredictionModel model) {
  Eigen::Matrix3d inertia = frame.inertia;
  if (model == OmegaPredictionModel::SRBM) {
    const Eigen::Matrix3d r_yaw = Rz3(frame.rpy(2));
    inertia = r_yaw * frame.nominal_inertia * r_yaw.transpose();
  }

  Eigen::Vector3d angular_impulse = frame.external_moment_impulse;
  if (model == OmegaPredictionModel::InertiaRate) {
    angular_impulse -=
        frame.inertia_dot_filtered * frame.omega * frame.integrated_dt;
  } else if (model == OmegaPredictionModel::InertiaRateRaw) {
    angular_impulse -=
        frame.inertia_dot_raw * frame.omega * frame.integrated_dt;
  }
  return frame.omega + inertia.inverse() * angular_impulse;
}

struct ExternalWrenchSample {
  Eigen::Vector3d force{Eigen::Vector3d::Zero()};
  Eigen::Vector3d moment{Eigen::Vector3d::Zero()};
  int contact_count{0};
};

ExternalWrenchSample computeActualExternalWrench(
    const mjModel *model, const mjData *data, const Eigen::Vector3d &com_W,
    int base_body_id, const Eigen::Vector3d &applied_push_force_W) {
  ExternalWrenchSample sample;
  for (int contact_id = 0; contact_id < data->ncon; ++contact_id) {
    const mjContact &contact = data->contact[contact_id];
    if (contact.geom[0] < 0 || contact.geom[1] < 0) {
      continue;
    }
    const int body0 = model->geom_bodyid[contact.geom[0]];
    const int body1 = model->geom_bodyid[contact.geom[1]];
    const bool body0_is_world = body0 == 0;
    const bool body1_is_world = body1 == 0;
    if (body0_is_world == body1_is_world) {
      continue;
    }

    mjtNum local_wrench[6] = {0, 0, 0, 0, 0, 0};
    mj_contactForce(model, data, contact_id, local_wrench);
    Eigen::Matrix3d contact_frame;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        contact_frame(row, col) = contact.frame[3 * row + col];
      }
    }
    const Eigen::Vector3d force_local(local_wrench[0], local_wrench[1],
                                      local_wrench[2]);
    const Eigen::Vector3d moment_local(local_wrench[3], local_wrench[4],
                                       local_wrench[5]);
    // mj_contactForce returns the wrench acting on geom[1].
    const double robot_wrench_sign = body1_is_world ? -1.0 : 1.0;
    const Eigen::Vector3d force_W =
        robot_wrench_sign * contact_frame.transpose() * force_local;
    const Eigen::Vector3d moment_W =
        robot_wrench_sign * contact_frame.transpose() * moment_local;
    const Eigen::Vector3d contact_pos_W(contact.pos[0], contact.pos[1],
                                        contact.pos[2]);
    sample.force += force_W;
    sample.moment += (contact_pos_W - com_W).cross(force_W) + moment_W;
    ++sample.contact_count;
  }

  if (applied_push_force_W.squaredNorm() > 0.0 && base_body_id >= 0) {
    const Eigen::Vector3d application_point_W(
        data->xipos[3 * base_body_id], data->xipos[3 * base_body_id + 1],
        data->xipos[3 * base_body_id + 2]);
    sample.force += applied_push_force_W;
    sample.moment +=
        (application_point_W - com_W).cross(applied_push_force_W);
  }
  return sample;
}

bool getEnvBool(const char *name, bool default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }

  const std::string text(value);
  if (text == "1" || text == "true" || text == "TRUE" || text == "on" ||
      text == "ON" || text == "yes" || text == "YES") {
    return true;
  }
  if (text == "0" || text == "false" || text == "FALSE" || text == "off" ||
      text == "OFF" || text == "no" || text == "NO") {
    return false;
  }
  return default_value;
}

double getEnvDouble(const char *name, double default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }

  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  return end != value ? parsed : default_value;
}

int getEnvInt(const char *name, int default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }

  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end != value ? static_cast<int>(parsed) : default_value;
}

std::string getEnvString(const char *name, const std::string &default_value) {
  const char *value = std::getenv(name);
  return value == nullptr ? default_value : std::string(value);
}

std::vector<double> parseEnvDoubleList(const std::string &text) {
  std::vector<double> values;
  if (text.empty()) {
    return values;
  }

  std::string normalized = text;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');
  std::istringstream input(normalized);
  double value = 0.0;
  while (input >> value) {
    values.push_back(value);
  }
  return values;
}

bool parseMpcStateWeights(const std::string &text,
                          Eigen::Matrix<double, 1, nx> &weights_out) {
  if (text.empty()) {
    return false;
  }

  std::string normalized = text;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');
  std::istringstream input(normalized);
  Eigen::Matrix<double, 1, nx> parsed;
  for (int i = 0; i < nx; ++i) {
    if (!(input >> parsed(i))) {
      return false;
    }
  }

  double extra = 0.0;
  if (input >> extra) {
    return false;
  }

  weights_out = parsed;
  return true;
}

void writeSummaryHeaderIfNeeded(const std::string &file_path) {
  std::ifstream test_in(file_path);
  const bool need_header = !test_in.good() || test_in.peek() == EOF;
  test_in.close();

  if (!need_header) {
    return;
  }

  std::ofstream out(file_path, std::ios::out);
  out << "exp_id,exp_name,use_variable_inertia,use_tau_bias,leg_mass_fraction,"
         "target_speed_x,target_speed_y,push_force,push_start_time,push_duration,stable_steps,"
         "fall_detected,fall_time,final_time,controller_mass,controller_leg_mass\n";
}

void applyMuJoCoLegMassFraction(mjModel *model, mjData *data,
                                double leg_mass_fraction) {
  constexpr double kMinScale = 1e-3;
  constexpr double kMaxLegMassFraction = 0.8;
  static constexpr std::array<const char *, 12> kLegBodyNames = {
      "Link_hip_l_roll",   "Link_hip_l_yaw",   "Link_hip_l_pitch",
      "Link_knee_l_pitch", "Link_ankle_l_pitch", "Link_ankle_l_roll",
      "Link_hip_r_roll",   "Link_hip_r_yaw",   "Link_hip_r_pitch",
      "Link_knee_r_pitch", "Link_ankle_r_pitch", "Link_ankle_r_roll"};

  static bool initialized = false;
  static std::vector<bool> is_leg_body;
  static std::vector<mjtNum> nominal_body_mass;
  static std::vector<std::array<mjtNum, 3>> nominal_body_inertia;
  static double nominal_total_mass = 0.0;
  static double nominal_leg_mass = 0.0;
  static double nominal_non_leg_mass = 0.0;

  if (!initialized) {
    is_leg_body.assign(model->nbody, false);
    nominal_body_mass.assign(model->nbody, 0.0);
    nominal_body_inertia.assign(model->nbody, {0.0, 0.0, 0.0});

    for (size_t idx = 0; idx < kLegBodyNames.size(); ++idx) {
      const int body_id = mj_name2id(model, mjOBJ_BODY, kLegBodyNames[idx]);
      if (body_id < 0) {
        std::cerr << "MuJoCo body not found for leg mass fraction scaling: "
                  << kLegBodyNames[idx] << std::endl;
        continue;
      }
      is_leg_body[body_id] = true;
    }

    for (int body_id = 1; body_id < model->nbody; ++body_id) {
      nominal_body_mass[body_id] = model->body_mass[body_id];
      for (int axis = 0; axis < 3; ++axis) {
        nominal_body_inertia[body_id][axis] =
            model->body_inertia[3 * body_id + axis];
      }

      nominal_total_mass += nominal_body_mass[body_id];
      if (is_leg_body[body_id]) {
        nominal_leg_mass += nominal_body_mass[body_id];
      }
    }
    nominal_non_leg_mass = nominal_total_mass - nominal_leg_mass;
    initialized = true;
  }

  const double clamped_fraction =
      std::clamp(leg_mass_fraction, 0.0, kMaxLegMassFraction);
  const double desired_leg_mass = clamped_fraction * nominal_total_mass;
  const double leg_scale =
      nominal_leg_mass > 1e-9
          ? std::max(desired_leg_mass / nominal_leg_mass, kMinScale)
          : 1.0;
  const double non_leg_scale =
      nominal_non_leg_mass > 1e-9
          ? std::max((nominal_total_mass - leg_scale * nominal_leg_mass) /
                         nominal_non_leg_mass,
                     kMinScale)
          : 1.0;

  for (int body_id = 1; body_id < model->nbody; ++body_id) {
    const double scale = is_leg_body[body_id] ? leg_scale : non_leg_scale;
    model->body_mass[body_id] = nominal_body_mass[body_id] * scale;
    for (int axis = 0; axis < 3; ++axis) {
      model->body_inertia[3 * body_id + axis] =
          nominal_body_inertia[body_id][axis] * scale;
    }
  }

  mj_setConst(model, data);
  mj_resetData(model, data);
}

void applyMuJoCoLegInertiaScale(mjModel *model, mjData *data,
                                double leg_scale) {
  constexpr double kMinScale = 1e-3;
  static constexpr std::array<const char *, 12> kLegBodyNames = {
      "Link_hip_l_roll",   "Link_hip_l_yaw",   "Link_hip_l_pitch",
      "Link_knee_l_pitch", "Link_ankle_l_pitch", "Link_ankle_l_roll",
      "Link_hip_r_roll",   "Link_hip_r_yaw",   "Link_hip_r_pitch",
      "Link_knee_r_pitch", "Link_ankle_r_pitch", "Link_ankle_r_roll"};

  static bool initialized = false;
  static std::vector<bool> is_leg_body;
  static std::vector<mjtNum> nominal_body_mass;
  static std::vector<std::array<mjtNum, 3>> nominal_body_inertia;

  if (!initialized) {
    is_leg_body.assign(model->nbody, false);
    nominal_body_mass.assign(model->nbody, 0.0);
    nominal_body_inertia.assign(model->nbody, {0.0, 0.0, 0.0});

    for (const char *body_name : kLegBodyNames) {
      const int body_id = mj_name2id(model, mjOBJ_BODY, body_name);
      if (body_id < 0) {
        std::cerr << "MuJoCo body not found for leg lambda scaling: "
                  << body_name << std::endl;
        continue;
      }
      is_leg_body[body_id] = true;
    }

    for (int body_id = 1; body_id < model->nbody; ++body_id) {
      nominal_body_mass[body_id] = model->body_mass[body_id];
      for (int axis = 0; axis < 3; ++axis) {
        nominal_body_inertia[body_id][axis] =
            model->body_inertia[3 * body_id + axis];
      }
    }
    initialized = true;
  }

  const double clamped_leg_scale = std::max(leg_scale, kMinScale);
  for (int body_id = 1; body_id < model->nbody; ++body_id) {
    const double scale = is_leg_body[body_id] ? clamped_leg_scale : 1.0;
    model->body_mass[body_id] = nominal_body_mass[body_id] * scale;
    for (int axis = 0; axis < 3; ++axis) {
      model->body_inertia[3 * body_id + axis] =
          nominal_body_inertia[body_id][axis] * scale;
    }
  }

  mj_setConst(model, data);
  mj_resetData(model, data);
}
} // namespace

int main(int argc, char **argv) {
  if (mj_model == nullptr || mj_data == nullptr) {
    std::cerr << "Failed to load MuJoCo scene " << scene_model_path << ": "
              << error << std::endl;
    return 1;
  }
  // Experiment selector. The user manually switches this flag.
  int8_t exp = static_cast<int8_t>(getEnvInt("ODC_EXP", 1));
  // exp = 1: leg mass fraction sweep
  // exp = 2: speed sweep
  // exp = 3: tau_bias ablation
  // exp = 4: disturbance recovery

  // Shared experiment toggles:
  bool use_variable_inertia_model =
	  false; // false -> SRBM, true -> VICM/VIBM
  bool use_tau_bias_feedforward =
	  true; // used mainly for exp = 3 (false -> ablation)

  Pin_KinDyn kinDynSolver("../models/AzureLoong.urdf");
  const double nominal_leg_mass_fraction = kinDynSolver.getNominalLegMassFraction();

  // User-editable experiment parameters:
//  double leg_mass_fraction = nominal_leg_mass_fraction; // exp = 1, share of total mass in legs, 0.0 ~ 0.8
  double leg_mass_fraction = 0.5;
  double leg_lambda_scale = 1.0;
  double target_speed_x = 1.5;   // exp = 2 / 4
  double target_speed_y = 0.0;   // exp = 2 / 4
  double push_force = 0.0;     // exp = 1 / 4, world-frame push along push_dir_W
  double push_start_time = 6.0;  // exp = 1 / 4
  double push_duration = 0.15;   // exp = 1 / 4
  std::string push_trigger_mode = "time";
  double push_trigger_phi = 0.5;
  double tau_bias_scale = 1.0;
  bool print_variable_inertia = false;
  double ig_print_interval = 0.5; // seconds
  bool print_fr_ff = true;
  double fr_print_interval = 0.05; // seconds
  Eigen::Vector3d push_dir_W(1.0, 0.0, 0.0);
  std::string exp_name = "exp1_leg_fraction_sweep";

  switch (exp) {
  case 1:
    exp_name = "exp1_leg_fraction_sweep";
    // target_speed_x = 0.0;
    // target_speed_y = 0.25;
    break;
  case 2:
    exp_name = "exp2_speed_sweep";
    leg_mass_fraction = nominal_leg_mass_fraction;
    push_force = 0.0;
    break;
  case 3:
    exp_name = "exp3_tau_bias_ablation";
    leg_mass_fraction = nominal_leg_mass_fraction;
    // target_speed_x = 0.0;
    // target_speed_y = 0.25;
    use_tau_bias_feedforward = false;
    push_force = 0.0;
    break;
  case 4:
    exp_name = "exp4_disturbance_recovery";
    leg_mass_fraction = nominal_leg_mass_fraction;
    // target_speed_x = 0.0;
    // target_speed_y = 0.25;
    break;
  default:
    std::cerr << "Unsupported experiment id: " << static_cast<int>(exp)
              << std::endl;
    return 1;
  }

  use_variable_inertia_model =
      getEnvBool("ODC_USE_VICM", use_variable_inertia_model);
  use_tau_bias_feedforward =
      getEnvBool("ODC_USE_TAU_BIAS", use_tau_bias_feedforward);
  leg_mass_fraction =
      getEnvDouble("ODC_LEG_MASS_FRACTION", leg_mass_fraction);
  const bool use_leg_lambda_scale =
      getEnvBool("ODC_USE_LEG_LAMBDA_SCALE", false);
  leg_lambda_scale =
      getEnvDouble("ODC_LEG_LAMBDA_SCALE", leg_lambda_scale);
  target_speed_x = getEnvDouble("ODC_TARGET_SPEED_X", target_speed_x);
  target_speed_y = getEnvDouble("ODC_TARGET_SPEED_Y", target_speed_y);
  push_force = getEnvDouble("ODC_PUSH_FORCE", push_force);
  push_start_time = getEnvDouble("ODC_PUSH_START_TIME", push_start_time);
  push_duration = getEnvDouble("ODC_PUSH_DURATION", push_duration);
  push_trigger_mode = getEnvString("ODC_PUSH_TRIGGER_MODE", push_trigger_mode);
  const bool push_phase_trigger_enabled =
      getEnvBool("ODC_PUSH_PHASE_TRIGGER", push_trigger_mode == "phase");
  push_trigger_phi = getEnvDouble("ODC_PUSH_TRIGGER_PHI", push_trigger_phi);
  const int push_recovery_stop_steps =
      std::max(0, getEnvInt("ODC_PUSH_RECOVERY_STOP_STEPS", 0));
  push_dir_W(0) = getEnvDouble("ODC_PUSH_DIR_X", push_dir_W(0));
  push_dir_W(1) = getEnvDouble("ODC_PUSH_DIR_Y", push_dir_W(1));
  push_dir_W(2) = getEnvDouble("ODC_PUSH_DIR_Z", push_dir_W(2));
  if (push_dir_W.norm() < 1e-9) {
    std::cerr << "ODC_PUSH_DIR vector is near zero. Falling back to +x."
              << std::endl;
    push_dir_W = Eigen::Vector3d(1.0, 0.0, 0.0);
  }
  tau_bias_scale = getEnvDouble("ODC_TAU_BIAS_SCALE", tau_bias_scale);
  print_variable_inertia =
      getEnvBool("ODC_PRINT_IG", print_variable_inertia);
  print_fr_ff = getEnvBool("ODC_PRINT_FR_FF", print_fr_ff);
  fr_print_interval =
      getEnvDouble("ODC_FR_PRINT_INTERVAL", fr_print_interval);
  const bool print_mpc_timing = getEnvBool("ODC_PRINT_MPC_TIMING", true);
  const double mpc_timing_print_interval =
      getEnvDouble("ODC_MPC_TIMING_PRINT_INTERVAL", 1.0);
  const bool print_gait_switch = getEnvBool("ODC_PRINT_GAIT_SWITCH", false);
  const bool sensor_noise_enabled =
      getEnvBool("ODC_SENSOR_NOISE_ENABLE", false);
  const uint32_t sensor_noise_seed = static_cast<uint32_t>(
      std::max(0, getEnvInt("ODC_SENSOR_NOISE_SEED", 1)));
  const double noise_base_pos_std =
      std::max(0.0, getEnvDouble("ODC_NOISE_BASE_POS_STD", 0.0));
  const double noise_base_rpy_std =
      std::max(0.0, getEnvDouble("ODC_NOISE_BASE_RPY_STD", 0.0));
  const double noise_base_vel_std =
      std::max(0.0, getEnvDouble("ODC_NOISE_BASE_VEL_STD", 0.0));
  const double noise_base_omega_std =
      std::max(0.0, getEnvDouble("ODC_NOISE_BASE_OMEGA_STD", 0.0));
  const double noise_joint_pos_std =
      std::max(0.0, getEnvDouble("ODC_NOISE_JOINT_POS_STD", 0.0));
  const double noise_joint_vel_std =
      std::max(0.0, getEnvDouble("ODC_NOISE_JOINT_VEL_STD", 0.0));
  const double noise_foot_force_std =
      std::max(0.0, getEnvDouble("ODC_NOISE_FOOT_FORCE_STD", 0.0));
  const bool headless = getEnvBool("ODC_HEADLESS", false);
  const bool snapshot_enabled = getEnvBool("ODC_SNAPSHOT_ENABLE", false);
  const std::string snapshot_dir =
      getEnvString("ODC_SNAPSHOT_DIR", "../record/walking_snapshots");
  const std::string snapshot_prefix =
      getEnvString("ODC_SNAPSHOT_PREFIX", "walking_snapshot");
  const double snapshot_start_time =
      getEnvDouble("ODC_SNAPSHOT_START_TIME", 10.0);
  const double snapshot_interval =
      std::max(getEnvDouble("ODC_SNAPSHOT_INTERVAL", 1.0), mj_model->opt.timestep);
  const std::vector<double> snapshot_times =
      parseEnvDoubleList(getEnvString("ODC_SNAPSHOT_TIMES", ""));
  const int snapshot_count =
      snapshot_times.empty()
          ? std::max(0, getEnvInt("ODC_SNAPSHOT_COUNT", 10))
          : static_cast<int>(snapshot_times.size());
  const bool snapshot_exit_after_capture =
      getEnvBool("ODC_SNAPSHOT_EXIT_AFTER_CAPTURE", false);
  const bool use_linear_inertia_prediction =
      getEnvBool("ODC_PREDICT_IG_LINEAR", false);
  const bool use_linear_tau_dynamics =
      getEnvBool("ODC_LINEAR_TAU_DYNAMICS", use_linear_inertia_prediction);
  const bool use_discrete_momentum_dynamics =
      getEnvBool("ODC_DISCRETE_MOMENTUM_DYNAMICS", false);
  const bool use_discrete_momentum_q_preview =
      getEnvBool("ODC_DISCRETE_MOMENTUM_Q_PREVIEW", true);
  const bool use_ircmpc_rolling_inertia =
      getEnvBool("ODC_IRCMPC_ROLLING_INERTIA", false);
  const bool use_tau_phase_gate = getEnvBool("ODC_TAU_PHASE_GATE", false);
  const double tau_phase_gate_min = getEnvDouble("ODC_TAU_PHASE_MIN", 0.2);
  const double tau_phase_gate_max = getEnvDouble("ODC_TAU_PHASE_MAX", 0.8);
  const bool use_sine_speed_profile = getEnvBool("ODC_SINE_SPEED", false);
  const double sine_vx_base = getEnvDouble("ODC_SINE_VX_BASE", target_speed_x);
  const double sine_vx_amp = getEnvDouble("ODC_SINE_VX_AMP", 0.25);
  const double sine_vx_period = getEnvDouble("ODC_SINE_VX_PERIOD", 4.0);
  const double sine_start_time = getEnvDouble("ODC_SINE_START_TIME", 4.0);
  const bool use_step_speed_profile = getEnvBool("ODC_STEP_SPEED", false);
  const double step_speed_time = getEnvDouble("ODC_STEP_SPEED_TIME", 10.0);
  const double step_vx_1 = getEnvDouble("ODC_STEP_VX_1", 1.0);
  const double step_vx_2 = getEnvDouble("ODC_STEP_VX_2", 1.8);
  const double step_vy_1 = getEnvDouble("ODC_STEP_VY_1", target_speed_y);
  const double step_vy_2 = getEnvDouble("ODC_STEP_VY_2", step_vy_1);
  const double step_speed_ramp_time =
      getEnvDouble("ODC_STEP_SPEED_RAMP_TIME", 3.0);
  const bool use_sine_turn_profile = getEnvBool("ODC_SINE_TURN", false);
  const double sine_wz_base = getEnvDouble("ODC_SINE_WZ_BASE", 0.0);
  const double sine_wz_amp = getEnvDouble("ODC_SINE_WZ_AMP", 0.4);
  const double sine_wz_period = getEnvDouble("ODC_SINE_WZ_PERIOD", 4.0);
  const double sine_wz_start_time =
      getEnvDouble("ODC_SINE_WZ_START_TIME", sine_start_time);
  const bool log_prediction_error =
      getEnvBool("ODC_LOG_PREDICTION_ERROR", false);
  const double prediction_ig_dot_filter_tau = std::max(
      0.0, getEnvDouble("ODC_PREDICTION_IG_DOT_FILTER_TAU", 0.01));
  const double gait_switch_force_threshold =
      getEnvDouble("ODC_GAIT_SWITCH_FORCE_THRESHOLD", 100.0);
  StairTerrainProfile stair_terrain;
  stair_terrain.enabled = getEnvBool("ODC_STAIR_MODE", false);
  stair_terrain.first_riser_x =
      getEnvDouble("ODC_STAIR_FIRST_RISER_X", 0.5);
  stair_terrain.tread_depth =
      getEnvDouble("ODC_STAIR_TREAD_DEPTH", 0.5);
  stair_terrain.riser_height =
      getEnvDouble("ODC_STAIR_RISER_HEIGHT", 0.15);
  stair_terrain.max_height =
      getEnvDouble("ODC_STAIR_MAX_HEIGHT", 1.5);
  const double gait_swing_time = getEnvDouble(
      "ODC_TSWING", stair_terrain.enabled ? 0.55 : 0.45);
  const bool use_stair_contact_preview =
      stair_terrain.enabled && getEnvBool("ODC_STAIR_CONTACT_PREVIEW", true);
  const double gait_min_touchdown_phase = getEnvDouble(
      "ODC_GAIT_MIN_TOUCHDOWN_PHASE", stair_terrain.enabled ? 0.85 : 0.6);
  const bool use_touchdown_position_gate = stair_terrain.enabled &&
      getEnvBool("ODC_GAIT_TOUCHDOWN_POSITION_GATE", true);
  const double touchdown_position_tolerance =
      getEnvDouble("ODC_GAIT_TOUCHDOWN_POSITION_TOLERANCE", 0.18);
  const double touchdown_height_tolerance =
      getEnvDouble("ODC_GAIT_TOUCHDOWN_HEIGHT_TOLERANCE", 0.15);
  const double torque_limit_scale =
      getEnvDouble("ODC_TORQUE_LIMIT_SCALE", 1.0);
  const double walk_leg_pd_scale =
      getEnvDouble("ODC_WALK_LEG_PD_SCALE", 1.0);
  const std::string gait_switch_force_source =
      getEnvString("ODC_GAIT_SWITCH_FORCE_SOURCE", "touch");
  const std::string mpc_weight_preset =
      getEnvString("ODC_MPC_WEIGHT_PRESET", "baseline");
  const std::string mpc_l_diag_override =
      getEnvString("ODC_MPC_L_DIAG", "");
  Eigen::Matrix<double, 1, nx> mpc_l_diag_override_values;
  const bool has_mpc_l_diag_override =
      parseMpcStateWeights(mpc_l_diag_override, mpc_l_diag_override_values);
  if (!mpc_l_diag_override.empty() && !has_mpc_l_diag_override) {
    std::cerr << "Invalid ODC_MPC_L_DIAG. Expected " << nx
              << " comma- or space-separated values. Falling back to preset "
              << mpc_weight_preset << std::endl;
  }

  const std::string default_controller_label =
      use_variable_inertia_model
          ? (use_tau_bias_feedforward ? "VICM_tau" : "VICM_no_tau")
          : "SRBM";
  const std::string controller_label =
      getEnvString("ODC_RUN_LABEL", default_controller_label);
  const bool isolate_experiment_outputs =
      getEnvBool("ODC_ISOLATE_EXPERIMENT_OUTPUTS", false);

  const double requested_leg_mass_fraction = leg_mass_fraction;
  leg_mass_fraction = std::clamp(leg_mass_fraction, 0.0, 0.8);
  if (std::abs(leg_mass_fraction - requested_leg_mass_fraction) > 1e-9) {
    std::cerr << "Requested leg_mass_fraction=" << requested_leg_mass_fraction
              << " is outside [0.0, 0.8]. Clamped to " << leg_mass_fraction
              << std::endl;
  }

  const std::string output_suffix =
      isolate_experiment_outputs ? "_" + controller_label : "";
  const std::string exp_tag = "../record/exp" +
                              std::to_string(static_cast<int>(exp)) +
                              output_suffix;
  const std::string summary_path =
      "../record/exp_summary" + output_suffix + ".csv";
  const std::string fr_ff_path =
      "../record/fr_ff_exp" + std::to_string(static_cast<int>(exp)) + "_" +
      controller_label + "_lf" + std::to_string(leg_mass_fraction) + ".csv";
  const std::string pred_error_path =
      "../record/pred_error_exp" + std::to_string(static_cast<int>(exp)) + "_" +
      controller_label + "_lf" + std::to_string(leg_mass_fraction) + ".csv";
  const std::string mpc_horizon_path =
      "../record/mpc_horizon_exp" +
      std::to_string(static_cast<int>(exp)) + "_" + controller_label + "_lf" +
      std::to_string(leg_mass_fraction) + ".csv";

  if (use_leg_lambda_scale) {
    applyMuJoCoLegInertiaScale(mj_model, mj_data, leg_lambda_scale);
  } else {
    applyMuJoCoLegMassFraction(mj_model, mj_data, leg_mass_fraction);
  }

  writeSummaryHeaderIfNeeded(summary_path);
  DataLogger logger(exp_tag + "_datalog.log");
  std::ofstream trace_file(exp_tag + "_trace.csv", std::ios::out);
  std::ofstream fr_ff_file(fr_ff_path, std::ios::out);
  std::ofstream pred_error_file;
  std::ofstream mpc_horizon_file;
  if (log_prediction_error) {
    pred_error_file.open(pred_error_path, std::ios::out);
    mpc_horizon_file.open(mpc_horizon_path, std::ios::out);
  }
  std::ofstream summary_file(summary_path, std::ios::app);

  trace_file
      << "time,exp_id,use_variable_inertia,use_tau_bias,leg_mass_fraction,"
         "target_speed_x,target_speed_y,push_active,push_force,step_count,gait_phase,leg_state,"
         "base_x,base_y,base_z,roll,pitch,yaw,yaw_ref,vx,vy,vz,wx,wy,wz,"
         "vx_ref,vy_ref,wz_ref,vel_track_error,torso_angle_error,tau_bias_norm,"
           "tau_mpc_x,tau_mpc_y,tau_mpc_z,tau_mpc_norm,"
	         "tau_idot_x,tau_idot_y,tau_idot_z,tau_idot_norm,"
	         "tau_gyro_x,tau_gyro_y,tau_gyro_z,tau_gyro_norm,"
	         "mpc_qp_status,mpc_qp_nwsr,wbc_qp_status,wbc_qp_nwsr,"
	         "wbc_delta_fr_norm,"
	         "controller_mass,controller_leg_mass,fLz_touch,fRz_touch,"
         "fLz_contact_raw,fRz_contact_raw,fLz_contact,fRz_contact,"
	         "fLz_xml_touch,fRz_xml_touch,FLest_z,FRest_z,fall_detected,"
             "push_triggered,push_actual_start,recovery_steps,motion_state,"
             "stair_mode,terrain_height,base_height_ref,"
             "swing_target_x,swing_target_y,swing_target_z,"
             "left_foot_x,left_foot_z,right_foot_x,right_foot_z\n";

  fr_ff_file
      << "time,controller_label,exp_id,use_variable_inertia,use_tau_bias,"
         "leg_mass_fraction,target_speed_x,target_speed_y,push_active,"
         "push_force,step_count,gait_phase,leg_state,"
         "l_fx,l_fy,l_fz,l_tx,l_ty,l_tz,r_fx,r_fy,r_fz,r_tx,r_ty,r_tz,"
         "base_x,base_y,base_z,roll,pitch,yaw,vx,vy,vz,vel_track_error,"
         "torso_angle_error,fall_detected\n";
  if (log_prediction_error) {
    pred_error_file
        << "time,dt,controller_label,exp_id,leg_mass_fraction,"
           "start_roll,start_pitch,start_yaw,start_wx,start_wy,start_wz,"
           "nominal_i00,nominal_i01,nominal_i02,nominal_i10,nominal_i11,nominal_i12,nominal_i20,nominal_i21,nominal_i22,"
           "inertia_i00,inertia_i01,inertia_i02,inertia_i10,inertia_i11,inertia_i12,inertia_i20,inertia_i21,inertia_i22,"
           "idot_filtered_i00,idot_filtered_i01,idot_filtered_i02,idot_filtered_i10,idot_filtered_i11,idot_filtered_i12,idot_filtered_i20,idot_filtered_i21,idot_filtered_i22,"
           "idot_raw_i00,idot_raw_i01,idot_raw_i02,idot_raw_i10,idot_raw_i11,idot_raw_i12,idot_raw_i20,idot_raw_i21,idot_raw_i22,"
           "wz_ref,actual_wx,actual_wy,actual_wz,"
           "srbm_pred_wx,srbm_pred_wy,srbm_pred_wz,"
           "vi_pred_wx,vi_pred_wy,vi_pred_wz,"
           "ir_pred_wx,ir_pred_wy,ir_pred_wz,"
           "ir_nf_pred_wx,ir_nf_pred_wy,ir_nf_pred_wz,"
           "srbm_err_wx,srbm_err_wy,srbm_err_wz,srbm_err_norm,"
           "vi_err_wx,vi_err_wy,vi_err_wz,vi_err_norm,"
           "ir_err_wx,ir_err_wy,ir_err_wz,ir_err_norm,"
           "ir_nf_err_wx,ir_nf_err_wy,ir_nf_err_wz,ir_nf_err_norm,"
           "moment_impulse_x,moment_impulse_y,moment_impulse_z,"
           "mean_force_z,mean_contact_count,phi,leg_state\n";
    mpc_horizon_file
        << "origin_time,target_time,actual_time,horizon_steps,controller_label,"
           "exp_id,leg_mass_fraction,origin_phi,origin_leg_state,origin_wz_ref,"
           "start_wx,start_wy,start_wz,pred_wx,pred_wy,pred_wz,"
           "actual_wx,actual_wy,actual_wz,err_wx,err_wy,err_wz,err_norm,"
           "delta_wx,delta_wy,delta_wz,delta_norm\n";
  }

  UIctr uiController(mj_model, mj_data);
  MJ_Interface mj_interface(mj_model, mj_data);
  if (use_leg_lambda_scale) {
    kinDynSolver.applyLegInertiaScale(leg_lambda_scale);
  } else {
    kinDynSolver.applyLegMassFraction(leg_mass_fraction);
  }
  DataBus RobotState(kinDynSolver.model_nv);
  WBC_priority WBC_solv(kinDynSolver.model_nv, 18, 22, 0.7,
                        mj_model->opt.timestep);
  MPC MPC_solv(dt_200Hz);
  GaitScheduler gaitScheduler(gait_swing_time, mj_model->opt.timestep);
  gaitScheduler.useTouchSwitchForce = gait_switch_force_source != "estimate";
  gaitScheduler.FzThrehold = gait_switch_force_threshold;
  gaitScheduler.minTouchdownPhase = gait_min_touchdown_phase;
  gaitScheduler.useTouchdownPositionGate = use_touchdown_position_gate;
  gaitScheduler.touchdownPositionTolerance = touchdown_position_tolerance;
  gaitScheduler.touchdownHeightTolerance = touchdown_height_tolerance;
  std::cout << "[GaitScheduler] tSwing=" << gait_swing_time
            << " switch_force_source="
            << (gaitScheduler.useTouchSwitchForce ? "touch" : "estimate")
            << " switch_force_threshold=" << gaitScheduler.FzThrehold
            << " min_touchdown_phase=" << gaitScheduler.minTouchdownPhase
            << " touchdown_position_gate="
            << (gaitScheduler.useTouchdownPositionGate ? 1 : 0)
            << " touchdown_position_tolerance="
            << gaitScheduler.touchdownPositionTolerance
            << " touchdown_height_tolerance="
            << gaitScheduler.touchdownHeightTolerance
            << std::endl;
  if (use_leg_lambda_scale) {
    std::cout << "[LegScale] lambda=" << leg_lambda_scale << std::endl;
  }
  std::cout << "[Push] trigger_mode="
            << (push_phase_trigger_enabled ? "phase" : "time")
            << " warmup_time=" << push_start_time
            << " trigger_phi=" << push_trigger_phi
            << " duration=" << push_duration << std::endl;
  std::cout << "[VICM] linear_inertia_prediction="
            << (use_linear_inertia_prediction ? 1 : 0)
            << " linear_tau_dynamics=" << (use_linear_tau_dynamics ? 1 : 0)
            << " discrete_momentum_dynamics="
            << (use_discrete_momentum_dynamics ? 1 : 0)
            << " q_preview="
            << (use_discrete_momentum_q_preview ? 1 : 0)
            << " ircmpc_rolling_inertia="
            << (use_ircmpc_rolling_inertia ? 1 : 0)
            << " tau_phase_gate=" << (use_tau_phase_gate ? 1 : 0)
            << " tau_phase_window=[" << tau_phase_gate_min << ","
            << tau_phase_gate_max << "]"
            << std::endl;
  if (use_sine_speed_profile) {
    std::cout << "[SpeedProfile] sine vx_base=" << sine_vx_base
              << " vx_amp=" << sine_vx_amp
              << " period=" << sine_vx_period
              << " start_time=" << sine_start_time << std::endl;
  }
  if (use_step_speed_profile) {
    std::cout << "[SpeedProfile] step vx=" << step_vx_1 << "->" << step_vx_2
              << " vy=" << step_vy_1 << "->" << step_vy_2
              << " step_time=" << step_speed_time
              << " ramp_time=" << step_speed_ramp_time << std::endl;
  }
  if (use_sine_turn_profile) {
    std::cout << "[TurnProfile] sine wz_base=" << sine_wz_base
              << " wz_amp=" << sine_wz_amp
              << " period=" << sine_wz_period
              << " start_time=" << sine_wz_start_time << std::endl;
  }
  if (log_prediction_error) {
    std::cout << "[PredictionError] logging to " << pred_error_path
              << std::endl;
    std::cout << "[MpcHorizonPrediction] logging to " << mpc_horizon_path
              << std::endl;
  }
  if (snapshot_enabled) {
    std::filesystem::create_directories(snapshot_dir);
    std::cout << "[Snapshot] enabled dir=" << snapshot_dir
              << " prefix=" << snapshot_prefix
              << " start="
              << (snapshot_times.empty() ? snapshot_start_time
                                         : snapshot_times.front())
              << " interval=" << snapshot_interval
              << " count=" << snapshot_count
              << " hidden=" << (headless ? 1 : 0) << std::endl;
  }
  PVT_Ctr pvtCtr(mj_model->opt.timestep, "../common/joint_ctrl_config.json");
  pvtCtr.setTorqueLimitScale(torque_limit_scale);
  std::cout << "[PVT] torque_limit_scale=" << torque_limit_scale
            << " walk_leg_pd_scale=" << walk_leg_pd_scale << std::endl;
  FootPlacement footPlacement;
  JoyStickInterpreter jsInterp(mj_model->opt.timestep);

  const bool render_enabled = !headless || snapshot_enabled;
  if (render_enabled) {
    uiController.iniGLFW();
    uiController.enableTracking();
    uiController.createWindow(snapshot_enabled ? "Walking snapshots" : "Demo",
                              false, headless && snapshot_enabled);
  }

  const double stand_legLength = getEnvDouble("ODC_STAND_LEG_LENGTH", 1.05);
  const double foot_height = 0.07;
  const double stair_foot_contact_offset = getEnvDouble(
      "ODC_STAIR_FOOT_CONTACT_OFFSET", foot_height - 0.035);
  const double stair_landing_margin =
      getEnvDouble("ODC_STAIR_LANDING_MARGIN", 0.10);
  const int model_nv = kinDynSolver.model_nv;
  const double startSteppingTime = 2.0;
  const double startWalkingTime = 3.0;
  const double startupDoubleSupportDuration = 0.5;
  const double startupSpeedRampDuration = 1.5;
  const double simEndTime = getEnvDouble("ODC_SIM_END_TIME", 30.0);
  const double fallHeightThreshold = 0.55;
  const double fallAngleThreshold = 0.8;
  const int baseBodyId = mj_name2id(mj_model, mjOBJ_BODY, "base_link");
  const double igPrintInterval =
      std::max(ig_print_interval, mj_model->opt.timestep);

  RobotState.width_hips = getEnvDouble("ODC_WIDTH_HIPS", 0.209);
  footPlacement.kp_vx = getEnvDouble("ODC_FOOT_KP_VX", 0.1);
  footPlacement.kp_vy = getEnvDouble("ODC_FOOT_KP_VY", 0.03);
  footPlacement.kp_wz = getEnvDouble("ODC_FOOT_KP_WZ", 0.03);
  footPlacement.stepHeight = getEnvDouble(
      "ODC_FOOT_STEP_HEIGHT", stair_terrain.enabled ? 0.40 : 0.205);
  footPlacement.xOff_L = getEnvDouble("ODC_FOOT_X_OFFSET_L", -0.01);
  footPlacement.yOff_L = getEnvDouble("ODC_FOOT_Y_OFFSET_L", 0.01);
  footPlacement.zOff_W = getEnvDouble("ODC_FOOT_Z_OFFSET_W", -0.035);
  footPlacement.lookaheadTime =
      getEnvDouble("ODC_FOOT_LOOKAHEAD_TIME", -1.0);
  footPlacement.firstStepLateralBiasScale = 0.25;
  footPlacement.firstStepHeightScale = 0.6;
  footPlacement.legLength = stand_legLength;
  footPlacement.stairTerrain = stair_terrain;
  footPlacement.stairFootContactOffset = stair_foot_contact_offset;
  footPlacement.stairLandingMargin = stair_landing_margin;
  RobotState.stair_terrain = stair_terrain;
  RobotState.use_stair_contact_preview = use_stair_contact_preview;
  RobotState.stair_nominal_base_height = stand_legLength + foot_height;
  RobotState.stair_foot_contact_offset = stair_foot_contact_offset;
  std::cout << "[FootPlacement] kp_vx=" << footPlacement.kp_vx
            << " xOff_L=" << footPlacement.xOff_L
            << " lookahead_time="
            << (footPlacement.lookaheadTime > 1e-6
                    ? footPlacement.lookaheadTime
                    : gait_swing_time)
            << " gait_tSwing=" << gait_swing_time << std::endl;
  std::cout << "[Terrain] scene=" << scene_model_path
            << " stair_mode=" << (stair_terrain.enabled ? 1 : 0)
            << " first_riser_x=" << stair_terrain.first_riser_x
            << " tread_depth=" << stair_terrain.tread_depth
            << " riser_height=" << stair_terrain.riser_height
            << " max_height=" << stair_terrain.max_height
            << " landing_margin=" << stair_landing_margin
            << " contact_preview=" << (use_stair_contact_preview ? 1 : 0)
            << std::endl;

  mju_copy(mj_data->qpos, mj_model->key_qpos, mj_model->nq * 1);

  std::vector<double> motors_vel_des(model_nv - 6, 0.0);
  std::vector<double> motors_tau_des(model_nv - 6, 0.0);

  Eigen::Vector3d fe_l_pos_L_des = {-0.018, 0.113, -stand_legLength};
  Eigen::Vector3d fe_r_pos_L_des = {-0.018, -0.116, -stand_legLength};
  Eigen::Vector3d fe_l_eul_L_des = {-0.000, -0.008, -0.000};
  Eigen::Vector3d fe_r_eul_L_des = {0.000, -0.008, 0.000};
  Eigen::Matrix3d fe_l_rot_des = eul2Rot(fe_l_eul_L_des(0), fe_l_eul_L_des(1),
                                         fe_l_eul_L_des(2));
  Eigen::Matrix3d fe_r_rot_des = eul2Rot(fe_r_eul_L_des(0), fe_r_eul_L_des(1),
                                         fe_r_eul_L_des(2));

  Eigen::Vector3d hd_l_pos_L_des = {-0.02, 0.32, -0.159};
  Eigen::Vector3d hd_r_pos_L_des = {-0.02, -0.32, -0.159};
  Eigen::Vector3d hd_l_eul_L_des = {-1.253, 0.122, -1.732};
  Eigen::Vector3d hd_r_eul_L_des = {1.253, 0.122, 1.732};
  Eigen::Matrix3d hd_l_rot_des = eul2Rot(hd_l_eul_L_des(0), hd_l_eul_L_des(1),
                                         hd_l_eul_L_des(2));
  Eigen::Matrix3d hd_r_rot_des = eul2Rot(hd_r_eul_L_des(0), hd_r_eul_L_des(1),
                                         hd_r_eul_L_des(2));

  auto resLeg = kinDynSolver.computeInK_Leg(fe_l_rot_des, fe_l_pos_L_des,
                                            fe_r_rot_des, fe_r_pos_L_des);
  auto resHand = kinDynSolver.computeInK_Hand(hd_l_rot_des, hd_l_pos_L_des,
                                              hd_r_rot_des, hd_r_pos_L_des);
  Eigen::VectorXd qIniDes = Eigen::VectorXd::Zero(mj_model->nq, 1);
  qIniDes.block(7, 0, mj_model->nq - 7, 1) =
      resLeg.jointPosRes + resHand.jointPosRes;
  WBC_solv.setQini(qIniDes, RobotState.q);

  logger.addIterm("simTime", 1);
  logger.addIterm("exp_id", 1);
  logger.addIterm("use_vicm", 1);
  logger.addIterm("use_tau_bias", 1);
  logger.addIterm("leg_mass_fraction", 1);
  logger.addIterm("speed_ref_x", 1);
  logger.addIterm("speed_ref_y", 1);
  logger.addIterm("push_force_cmd", 1);
  logger.addIterm("gait_phase", 1);
  logger.addIterm("leg_state", 1);
  logger.addIterm("step_count", 1);
  logger.addIterm("vel_track_error", 1);
  logger.addIterm("torso_angle_error", 1);
  logger.addIterm("tau_bias_norm", 1);
  logger.addIterm("tau_mpc", 3);
  logger.addIterm("tau_mpc_norm", 1);
  logger.addIterm("tau_idot_omega", 3);
  logger.addIterm("tau_idot_omega_norm", 1);
  logger.addIterm("tau_gyro", 3);
  logger.addIterm("tau_gyro_norm", 1);
  logger.addIterm("mpc_qp_status", 1);
  logger.addIterm("mpc_qp_nwsr", 1);
	  logger.addIterm("wbc_qp_status", 1);
	  logger.addIterm("wbc_qp_nwsr", 1);
	  logger.addIterm("wbc_delta_fr_norm", 1);
	  logger.addIterm("fall_flag", 1);
  logger.addIterm("controller_mass", 1);
  logger.addIterm("controller_leg_mass", 1);
  logger.addIterm("motor_pos_des", model_nv - 6);
  logger.addIterm("motor_pos_cur", model_nv - 6);
  logger.addIterm("motor_vel_des", model_nv - 6);
  logger.addIterm("motor_vel_cur", model_nv - 6);
  logger.addIterm("motor_tor_des", model_nv - 6);
  logger.addIterm("rpyVal", 3);
  logger.addIterm("base_omega_W", 3);
  logger.addIterm("gpsVal", 3);
  logger.addIterm("base_vel", 3);
  logger.addIterm("dX_cal", 12);
  logger.addIterm("Ufe", 12);
  logger.addIterm("Xd", 12);
  logger.addIterm("X_cur", 12);
  logger.addIterm("X_cal", 12);
  logger.finishItermAdding();

  int MPC_count = 0;
  uint32_t stepCount = 0;
  bool legStateInitialized = false;
  DataBus::LegState lastLegState = DataBus::RSt;
  bool fallDetected = false;
  double fallTime = simEndTime;
  bool walkCommandInitialized = false;
  bool stepSpeedSecondCommanded = false;
  double nextIgPrintTime = 0.0;
  double nextFrPrintTime = 0.0;
  double nextMpcTimingPrintTime = 0.0;
  uint64_t mpcTimingSamples = 0;
  double mpcTimingWallMsSum = 0.0;
  double mpcTimingWallMsMax = 0.0;
  double mpcTimingQpMsSum = 0.0;
  double mpcTimingQpMsMax = 0.0;
  uint64_t mpcTimingTotalSamples = 0;
  double mpcTimingTotalWallMsSum = 0.0;
  double mpcTimingTotalWallMsMax = 0.0;
  double mpcTimingTotalQpMsSum = 0.0;
  double mpcTimingTotalQpMsMax = 0.0;
  OneStepPredictionFrame predictionFrame;
  Eigen::Matrix3d predictionIgDotFiltered = Eigen::Matrix3d::Zero();
  bool predictionIgDotFilterInitialized = false;
  std::deque<PendingMpcPrediction> pendingMpcPredictions;
  std::vector<CompletedMpcPrediction> completedMpcPredictions;
  completedMpcPredictions.reserve(
      static_cast<size_t>(simEndTime / dt_200Hz) * mpc_N);
  Eigen::Vector3d trueOmegaW = Eigen::Vector3d::Zero();
  int snapshot_index = 0;
  double next_snapshot_time =
      snapshot_times.empty() ? snapshot_start_time : snapshot_times.front();
  bool pushPhaseTriggered = false;
  double pushActualStartTime = push_start_time;
  double lastPushPhi = std::numeric_limits<double>::quiet_NaN();
  DataBus::LegState lastPushLegState = DataBus::DSt;
  bool recoveryStopCommanded = false;
  uint32_t recoveryTriggerStepCount = 0;
  uint32_t recoveryCompletedSteps = 0;
  std::mt19937 sensorNoiseRng(sensor_noise_seed);
  std::normal_distribution<double> standardNormal(0.0, 1.0);
  const auto sampleNoise = [&](double stddev) {
    return sensor_noise_enabled && stddev > 0.0
               ? stddev * standardNormal(sensorNoiseRng)
               : 0.0;
  };

  std::cout << std::fixed << std::setprecision(8)
            << "[Experiment uncertainty] sensor_noise="
            << (sensor_noise_enabled ? 1 : 0)
            << " seed=" << sensor_noise_seed
            << " base_pos_std=" << noise_base_pos_std
            << " base_rpy_std=" << noise_base_rpy_std
            << " base_vel_std=" << noise_base_vel_std
            << " base_omega_std=" << noise_base_omega_std
            << " joint_pos_std=" << noise_joint_pos_std
            << " joint_vel_std=" << noise_joint_vel_std
            << " foot_force_std=" << noise_foot_force_std << std::endl;

  mjtNum simstart = mj_data->time;
  double simTime = mj_data->time;
  while (headless || !glfwWindowShouldClose(uiController.window)) {
    simstart = mj_data->time;
    while (mj_data->time - simstart < 1.0 / 60.0 && uiController.runSim) {
      const bool pushEnabled = push_force > 0.0 && push_duration > 0.0;
      const bool pushEventEnabled =
          push_duration > 0.0 &&
          (push_force > 0.0 || push_recovery_stop_steps > 0);
      const bool pushReady =
          !push_phase_trigger_enabled || pushPhaseTriggered;
      const bool pushActive =
          pushEnabled && pushReady && mj_data->time >= pushActualStartTime &&
          mj_data->time < (pushActualStartTime + push_duration);
      Eigen::Vector3d appliedPushForceW = Eigen::Vector3d::Zero();
      for (int i = 0; i < 6; ++i) {
        mj_data->xfrc_applied[6 * baseBodyId + i] = 0.0;
      }
      if (pushActive) {
        appliedPushForceW = push_force * push_dir_W.normalized();
        for (int axis = 0; axis < 3; ++axis) {
          mj_data->xfrc_applied[6 * baseBodyId + axis] = appliedPushForceW(axis);
        }
      }

      mj_step(mj_model, mj_data);
      simTime = mj_data->time;

      mj_interface.updateSensorValues();
      mj_interface.dataBusWrite(RobotState);
      trueOmegaW = RobotState.dq.block<3, 1>(3, 0);
      // Use the exact scaled initial configuration to define the fixed SRBM
      // inertia for every paired trial; inject measurement noise afterwards.
      if (sensor_noise_enabled && MPC_solv.hasCalibratedNominalInertia()) {
        for (int axis = 0; axis < 3; ++axis) {
          RobotState.basePos[axis] += sampleNoise(noise_base_pos_std);
          RobotState.rpy[axis] += sampleNoise(noise_base_rpy_std);
          RobotState.baseLinVel[axis] += sampleNoise(noise_base_vel_std);
          RobotState.baseAngVel[axis] += sampleNoise(noise_base_omega_std);
        }
        for (double &position : RobotState.motors_pos_cur) {
          position += sampleNoise(noise_joint_pos_std);
        }
        for (double &velocity : RobotState.motors_vel_cur) {
          velocity += sampleNoise(noise_joint_vel_std);
        }
        RobotState.fL[2] =
            std::max(0.0, RobotState.fL[2] + sampleNoise(noise_foot_force_std));
        RobotState.fR[2] =
            std::max(0.0, RobotState.fR[2] + sampleNoise(noise_foot_force_std));
        RobotState.foot_contact_fz_l = RobotState.fL[2];
        RobotState.foot_contact_fz_r = RobotState.fR[2];
        RobotState.updateQ();
      }

      double command_speed_x = target_speed_x;
      double command_speed_y = target_speed_y;
      double command_wz = 0.0;
      if (use_sine_speed_profile) {
        command_speed_x = sine_vx_base;
        if (simTime >= sine_start_time && sine_vx_period > 1e-6) {
          const double phase =
              6.283185307179586 * (simTime - sine_start_time) / sine_vx_period;
          command_speed_x = sine_vx_base + sine_vx_amp * std::sin(phase);
        }
      }
      if (use_step_speed_profile) {
        command_speed_x = (simTime >= step_speed_time) ? step_vx_2 : step_vx_1;
        command_speed_y = (simTime >= step_speed_time) ? step_vy_2 : step_vy_1;
      }
      if (use_sine_turn_profile) {
        command_wz = sine_wz_base;
        if (simTime >= sine_wz_start_time && sine_wz_period > 1e-6) {
          const double phase = 6.283185307179586 *
                               (simTime - sine_wz_start_time) /
                               sine_wz_period;
          command_wz = sine_wz_base + sine_wz_amp * std::sin(phase);
        }
      }
      if (recoveryStopCommanded) {
        command_speed_x = 0.0;
        command_speed_y = 0.0;
        command_wz = 0.0;
      }

      RobotState.exp_id = static_cast<int>(exp);
      RobotState.leg_mass_fraction = leg_mass_fraction;
      RobotState.use_variable_inertia_model = use_variable_inertia_model;
      RobotState.use_tau_bias_feedforward = use_tau_bias_feedforward;
      RobotState.use_linear_inertia_prediction = use_linear_inertia_prediction;
      RobotState.use_linear_tau_dynamics = use_linear_tau_dynamics;
      RobotState.use_discrete_momentum_dynamics =
          use_discrete_momentum_dynamics;
      RobotState.use_ircmpc_rolling_inertia =
          use_ircmpc_rolling_inertia;
      RobotState.tau_bias_scale = tau_bias_scale;
      RobotState.use_tau_phase_gate = use_tau_phase_gate;
      RobotState.tau_phase_gate_min = tau_phase_gate_min;
      RobotState.tau_phase_gate_max = tau_phase_gate_max;
      RobotState.target_speed_x = command_speed_x;
      RobotState.target_speed_y = command_speed_y;
      RobotState.push_force_cmd = pushActive ? push_force : 0.0;
      RobotState.push_start_time = push_start_time;
      RobotState.push_duration = push_duration;

      kinDynSolver.dataBusRead(RobotState);
      kinDynSolver.computeJ_dJ();
      kinDynSolver.computeDyn();
      kinDynSolver.dataBusWrite(RobotState);

      if (log_prediction_error && predictionFrame.valid && baseBodyId >= 0) {
        const Eigen::Vector3d trueComW(
            mj_data->subtree_com[3 * baseBodyId],
            mj_data->subtree_com[3 * baseBodyId + 1],
            mj_data->subtree_com[3 * baseBodyId + 2]);
        const ExternalWrenchSample wrenchSample = computeActualExternalWrench(
            mj_model, mj_data, trueComW, baseBodyId, appliedPushForceW);
        const double physicsDt = mj_model->opt.timestep;
        predictionFrame.external_force_impulse += wrenchSample.force * physicsDt;
        predictionFrame.external_moment_impulse += wrenchSample.moment * physicsDt;
        predictionFrame.integrated_dt += physicsDt;
        predictionFrame.contact_count_sum += wrenchSample.contact_count;
        ++predictionFrame.physics_samples;
      }

      if (print_variable_inertia && use_variable_inertia_model &&
          simTime >= nextIgPrintTime) {
        std::cout << std::fixed << std::setprecision(6)
                  << "[VICM Ig] t=" << simTime
                  << " leg_mass_fraction=" << leg_mass_fraction << "\n"
                  << RobotState.inertia << std::endl;
        nextIgPrintTime = simTime + igPrintInterval;
      }

      if (simTime > startWalkingTime) {
        if (!walkCommandInitialized) {
          jsInterp.setVxDesLPara(
              use_step_speed_profile ? step_vx_1
                                     : (use_sine_speed_profile ? sine_vx_base
                                                               : target_speed_x),
              startupSpeedRampDuration);
          jsInterp.setVyDesLPara(use_step_speed_profile ? step_vy_1
                                                        : target_speed_y,
                                 startupSpeedRampDuration);
          jsInterp.setWzDesLPara(
              use_sine_turn_profile ? sine_wz_base : 0.0,
              startupSpeedRampDuration);
          walkCommandInitialized = true;
        }
        if (use_sine_speed_profile && simTime >= sine_start_time) {
          jsInterp.setVxDesLPara(command_speed_x, mj_model->opt.timestep);
        }
        if (use_step_speed_profile && simTime >= step_speed_time &&
            !stepSpeedSecondCommanded) {
          jsInterp.setVxDesLPara(command_speed_x, step_speed_ramp_time);
          jsInterp.setVyDesLPara(command_speed_y, step_speed_ramp_time);
          stepSpeedSecondCommanded = true;
        }
        if (use_sine_turn_profile && simTime >= sine_wz_start_time) {
          jsInterp.setWzDesLPara(command_wz, mj_model->opt.timestep);
        }

        if (simTime > (startWalkingTime + startupDoubleSupportDuration)) {
          RobotState.motionState = DataBus::Walk;
        } else {
          RobotState.motionState = DataBus::Stand;
        }
      } else {
        jsInterp.setIniPos(RobotState.q(0), RobotState.q(1),
                           RobotState.base_rpy(2));
        RobotState.motionState = DataBus::Stand;
      }
      jsInterp.step();
      double stair_support_height = 0.0;
      if (stair_terrain.enabled) {
        const double left_height =
            stair_terrain.stepHeightAt(RobotState.fe_l_pos_W(0));
        const double right_height =
            stair_terrain.stepHeightAt(RobotState.fe_r_pos_W(0));
        if (RobotState.legState == DataBus::LSt) {
          stair_support_height = left_height;
        } else if (RobotState.legState == DataBus::RSt) {
          stair_support_height = right_height;
        } else {
          stair_support_height = 0.5 * (left_height + right_height);
        }
      }
      RobotState.stair_support_height = stair_support_height;
      RobotState.js_pos_des(2) =
          stand_legLength + foot_height + stair_support_height;
      jsInterp.dataBusWrite(RobotState);

      if (simTime >= startSteppingTime) {
        const DataBus::LegState legStateBefore = RobotState.legState;
        const double phiBeforeScheduler = RobotState.phi;
        gaitScheduler.dataBusRead(RobotState);
        gaitScheduler.step();
        gaitScheduler.dataBusWrite(RobotState);

        footPlacement.dataBusRead(RobotState);
        footPlacement.getSwingPos();
        footPlacement.dataBusWrite(RobotState);

        if (!legStateInitialized) {
          lastLegState = RobotState.legState;
          legStateInitialized = true;
        } else if (RobotState.legState != lastLegState) {
          ++stepCount;
          if (recoveryStopCommanded && stepCount > recoveryTriggerStepCount) {
            recoveryCompletedSteps = stepCount - recoveryTriggerStepCount;
          }
          if (print_gait_switch) {
            const double fl_est_z =
                RobotState.FL_est.size() > 2
                    ? RobotState.FL_est(2)
                    : std::numeric_limits<double>::quiet_NaN();
            const double fr_est_z =
                RobotState.FR_est.size() > 2
                    ? RobotState.FR_est(2)
                    : std::numeric_limits<double>::quiet_NaN();
            std::cout << std::fixed << std::setprecision(6)
                      << "[GaitSwitch] t=" << simTime
                      << " step=" << stepCount
                      << " before=" << legStateName(legStateBefore)
                      << " last=" << legStateName(lastLegState)
                      << " after=" << legStateName(RobotState.legState)
                      << " phi_before=" << phiBeforeScheduler
                      << " phi_after=" << RobotState.phi
                      << " fLz_touch=" << RobotState.fL[2]
                      << " fRz_touch=" << RobotState.fR[2]
                      << " FLest_z=" << fl_est_z
                      << " FRest_z=" << fr_est_z
                      << " source="
                      << (gaitScheduler.useTouchSwitchForce ? "touch"
                                                            : "estimate")
                      << " threshold=" << gaitScheduler.FzThrehold
                      << std::endl;
          }
          lastLegState = RobotState.legState;
        }
      }

      RobotState.step_count = stepCount;

      if (pushEventEnabled && push_phase_trigger_enabled && !pushPhaseTriggered &&
          simTime >= push_start_time) {
        const bool phaseStateValid =
            RobotState.motionState == DataBus::Walk &&
            (RobotState.legState == DataBus::LSt ||
             RobotState.legState == DataBus::RSt);
        if (phaseStateValid) {
          const double phiNow = RobotState.phi;
          const bool sameLegState = lastPushLegState == RobotState.legState;
          const bool crossedTriggerPhi =
              std::isfinite(lastPushPhi) && sameLegState &&
              lastPushPhi < push_trigger_phi && phiNow >= push_trigger_phi;
          if (crossedTriggerPhi) {
            pushPhaseTriggered = true;
            pushActualStartTime = simTime + mj_model->opt.timestep;
            if (push_recovery_stop_steps > 0) {
              recoveryStopCommanded = true;
              recoveryTriggerStepCount = stepCount;
              recoveryCompletedSteps = 0;
              jsInterp.setVxDesLPara(0.0, mj_model->opt.timestep);
              jsInterp.setVyDesLPara(0.0, mj_model->opt.timestep);
              jsInterp.setWzDesLPara(0.0, mj_model->opt.timestep);
            }
          }
          lastPushPhi = phiNow;
          lastPushLegState = RobotState.legState;
        } else {
          lastPushPhi = std::numeric_limits<double>::quiet_NaN();
          lastPushLegState = DataBus::DSt;
        }
      }

      MPC_count = MPC_count + 1;
      if (MPC_count > (dt_200Hz / dt - 1)) {
        if (log_prediction_error && !pendingMpcPredictions.empty()) {
          for (auto it = pendingMpcPredictions.begin();
               it != pendingMpcPredictions.end();) {
            if (it->target_time > simTime + 0.5 * dt_200Hz) {
              ++it;
              continue;
            }
            completedMpcPredictions.push_back(
                {*it, simTime, trueOmegaW});
            it = pendingMpcPredictions.erase(it);
          }
        }

        if (log_prediction_error && predictionFrame.valid) {
          const double prediction_dt = simTime - predictionFrame.time;
          if (prediction_dt > 0.0 && prediction_dt < 0.05) {
            const Eigen::Vector3d actual_omega = trueOmegaW;
            const Eigen::Vector3d srbm_pred =
                predictOmegaOneStep(predictionFrame,
                                    OmegaPredictionModel::SRBM);
            const Eigen::Vector3d vi_pred = predictOmegaOneStep(
                predictionFrame, OmegaPredictionModel::VariableInertia);
            const Eigen::Vector3d ir_pred = predictOmegaOneStep(
                predictionFrame, OmegaPredictionModel::InertiaRate);
            const Eigen::Vector3d ir_nf_pred = predictOmegaOneStep(
                predictionFrame, OmegaPredictionModel::InertiaRateRaw);
            const Eigen::Vector3d srbm_err = actual_omega - srbm_pred;
            const Eigen::Vector3d vi_err = actual_omega - vi_pred;
            const Eigen::Vector3d ir_err = actual_omega - ir_pred;
            const Eigen::Vector3d ir_nf_err = actual_omega - ir_nf_pred;
            const double meanForceZ = predictionFrame.integrated_dt > 0.0
                                          ? predictionFrame.external_force_impulse(2) /
                                                predictionFrame.integrated_dt
                                          : 0.0;
            const double meanContactCount = predictionFrame.physics_samples > 0
                                                ? static_cast<double>(predictionFrame.contact_count_sum) /
                                                      predictionFrame.physics_samples
                                                : 0.0;
            pred_error_file
                << std::fixed << std::setprecision(6) << simTime << ","
                << prediction_dt << "," << controller_label << ","
                << static_cast<int>(exp) << "," << leg_mass_fraction << ","
                << predictionFrame.rpy(0) << "," << predictionFrame.rpy(1)
                << "," << predictionFrame.rpy(2) << ","
                << predictionFrame.omega(0) << ","
                << predictionFrame.omega(1) << ","
                << predictionFrame.omega(2);
            const auto write_matrix = [&pred_error_file](
                                          const Eigen::Matrix3d &matrix) {
              for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                  pred_error_file << "," << matrix(row, col);
                }
              }
            };
            write_matrix(predictionFrame.nominal_inertia);
            write_matrix(predictionFrame.inertia);
            write_matrix(predictionFrame.inertia_dot_filtered);
            write_matrix(predictionFrame.inertia_dot_raw);
            pred_error_file
                << "," << RobotState.js_omega_des(2) << ","
                << actual_omega(0) << "," << actual_omega(1) << ","
                << actual_omega(2) << ","
                << srbm_pred(0) << "," << srbm_pred(1) << ","
                << srbm_pred(2) << ","
                << vi_pred(0) << "," << vi_pred(1) << "," << vi_pred(2) << ","
                << ir_pred(0) << "," << ir_pred(1) << "," << ir_pred(2) << ","
                << ir_nf_pred(0) << "," << ir_nf_pred(1) << ","
                << ir_nf_pred(2) << ","
                << srbm_err(0) << "," << srbm_err(1) << ","
                << srbm_err(2) << "," << srbm_err.norm() << ","
                << vi_err(0) << "," << vi_err(1) << "," << vi_err(2) << ","
                << vi_err.norm() << ","
                << ir_err(0) << "," << ir_err(1) << "," << ir_err(2) << ","
                << ir_err.norm() << ","
                << ir_nf_err(0) << "," << ir_nf_err(1) << ","
                << ir_nf_err(2) << "," << ir_nf_err.norm() << ","
                << predictionFrame.external_moment_impulse(0) << ","
                << predictionFrame.external_moment_impulse(1) << ","
                << predictionFrame.external_moment_impulse(2) << ","
                << meanForceZ << "," << meanContactCount << ","
                << predictionFrame.phi << ","
                << static_cast<int>(predictionFrame.leg_state) << "\n";
          }
        }

        const auto mpcWallStart = std::chrono::steady_clock::now();
        if (((use_discrete_momentum_dynamics &&
              use_discrete_momentum_q_preview) ||
             use_ircmpc_rolling_inertia) &&
            use_variable_inertia_model) {
          kinDynSolver.computeCentroidalInertiaHorizon(
              RobotState, mpc_N + 1, dt_200Hz);
        } else {
          RobotState.inertia_horizon.clear();
        }
        MPC_solv.dataBusRead(RobotState);
        MPC_solv.cal();
        MPC_solv.dataBusWrite(RobotState);
        if (log_prediction_error && MPC_solv.get_ENA()) {
          const auto &predictedStates =
              MPC_solv.getPredictedStateSequence();
          for (int horizon = 1; horizon <= mpc_N; ++horizon) {
            PendingMpcPrediction pending;
            pending.origin_time = simTime;
            pending.target_time = simTime + horizon * dt_200Hz;
            pending.horizon_steps = horizon;
            pending.origin_phi = RobotState.phi;
            pending.origin_leg_state = RobotState.legState;
            pending.origin_wz_ref = RobotState.js_omega_des(2);
            pending.start_omega = trueOmegaW;
            pending.predicted_omega =
                predictedStates.block<3, 1>((horizon - 1) * nx + 6, 0);
            pendingMpcPredictions.push_back(pending);
          }

          const Eigen::Matrix3d rawIgDot = RobotState.inertia_dot;
          if (prediction_ig_dot_filter_tau > 0.0 && rawIgDot.allFinite()) {
            const double alpha = dt_200Hz /
                                 (prediction_ig_dot_filter_tau + dt_200Hz);
            if (!predictionIgDotFilterInitialized) {
              predictionIgDotFiltered = rawIgDot;
              predictionIgDotFilterInitialized = true;
            } else {
              predictionIgDotFiltered +=
                  alpha * (rawIgDot - predictionIgDotFiltered);
            }
          } else {
            predictionIgDotFiltered = rawIgDot;
            predictionIgDotFilterInitialized = false;
          }
          predictionIgDotFiltered =
              0.5 * (predictionIgDotFiltered + predictionIgDotFiltered.transpose());
          predictionFrame = makePredictionFrame(
              RobotState, simTime, MPC_solv.getNominalInertia(),
              predictionIgDotFiltered, trueOmegaW);
        }
        const auto mpcWallEnd = std::chrono::steady_clock::now();
        const double mpcWallMs =
            std::chrono::duration<double, std::milli>(mpcWallEnd -
                                                      mpcWallStart)
                .count();
        if (MPC_solv.get_ENA()) {
          const double mpcQpMs = RobotState.qp_cpuTime_MPC * 1000.0;
          ++mpcTimingSamples;
          mpcTimingWallMsSum += mpcWallMs;
          mpcTimingWallMsMax = std::max(mpcTimingWallMsMax, mpcWallMs);
          mpcTimingQpMsSum += mpcQpMs;
          mpcTimingQpMsMax = std::max(mpcTimingQpMsMax, mpcQpMs);
          ++mpcTimingTotalSamples;
          mpcTimingTotalWallMsSum += mpcWallMs;
          mpcTimingTotalWallMsMax =
              std::max(mpcTimingTotalWallMsMax, mpcWallMs);
          mpcTimingTotalQpMsSum += mpcQpMs;
          mpcTimingTotalQpMsMax =
              std::max(mpcTimingTotalQpMsMax, mpcQpMs);
          if (print_mpc_timing && simTime >= nextMpcTimingPrintTime &&
              mpcTimingSamples > 0) {
            std::cout << std::fixed << std::setprecision(3)
                      << "[MPC timing] t=" << simTime
                      << " samples=" << mpcTimingSamples
                      << " avg_wall_ms="
                      << (mpcTimingWallMsSum /
                          static_cast<double>(mpcTimingSamples))
                      << " max_wall_ms=" << mpcTimingWallMsMax
                      << " avg_qp_ms="
                      << (mpcTimingQpMsSum /
                          static_cast<double>(mpcTimingSamples))
                      << " max_qp_ms=" << mpcTimingQpMsMax
                      << " last_wall_ms=" << mpcWallMs
                      << " last_qp_ms=" << mpcQpMs
                      << " horizon_steps=" << mpc_N << std::endl;
            mpcTimingSamples = 0;
            mpcTimingWallMsSum = 0.0;
            mpcTimingWallMsMax = 0.0;
            mpcTimingQpMsSum = 0.0;
            mpcTimingQpMsMax = 0.0;
            nextMpcTimingPrintTime =
                simTime + std::max(mpc_timing_print_interval, dt_200Hz);
          }
        }
        if (print_fr_ff && simTime >= nextFrPrintTime) {
          std::cout << std::fixed << std::setprecision(6)
                    << "[MPC Fr_ff] t=" << simTime
                    << " use_vicm=" << (use_variable_inertia_model ? 1 : 0)
                    << " use_tau_bias=" << (use_tau_bias_feedforward ? 1 : 0)
                    << " leg_mass_fraction=" << leg_mass_fraction
                    << " Fr_ff=" << RobotState.Fr_ff.transpose() << std::endl;
          nextFrPrintTime =
              simTime + std::max(fr_print_interval, mj_model->opt.timestep);
        }
        const double fr_vel_track_error = computeVelTrackError(RobotState);
        const double fr_torso_angle_error = computeTorsoAngleError(RobotState);
        const bool fr_fall_detected = detectFall(
            RobotState,
            fallHeightThreshold + RobotState.stair_support_height,
            fallAngleThreshold);

        fr_ff_file << std::fixed << std::setprecision(6) << simTime << ","
                   << controller_label << "," << static_cast<int>(exp) << ","
                   << (use_variable_inertia_model ? 1 : 0) << ","
                   << (use_tau_bias_feedforward ? 1 : 0) << ","
                   << leg_mass_fraction << "," << target_speed_x << ","
                   << target_speed_y << "," << (pushActive ? 1 : 0) << ","
                   << RobotState.push_force_cmd << "," << RobotState.step_count
                   << "," << RobotState.phi << ","
                   << static_cast<int>(RobotState.legState);
        for (int fr_idx = 0; fr_idx < RobotState.Fr_ff.size(); ++fr_idx) {
          fr_ff_file << "," << RobotState.Fr_ff(fr_idx);
        }
        fr_ff_file << "," << RobotState.q(0) << "," << RobotState.q(1) << ","
                   << RobotState.q(2) << "," << RobotState.base_rpy(0) << ","
                   << RobotState.base_rpy(1) << "," << RobotState.base_rpy(2)
                   << "," << RobotState.dq(0) << "," << RobotState.dq(1)
                   << "," << RobotState.dq(2) << ","
                   << fr_vel_track_error << "," << fr_torso_angle_error << ","
                   << (fr_fall_detected ? 1 : 0) << "\n";
        MPC_count = 0;
      }

      WBC_solv.dataBusRead(RobotState);
      WBC_solv.computeDdq(kinDynSolver);
      WBC_solv.computeTau();
      WBC_solv.dataBusWrite(RobotState);

      if (simTime <= startSteppingTime) {
        RobotState.motors_pos_des =
            eigen2std(resLeg.jointPosRes + resHand.jointPosRes);
        RobotState.motors_vel_des = motors_vel_des;
        RobotState.motors_tor_des = motors_tau_des;
      } else {
        MPC_solv.enable();
        Eigen::Matrix<double, 1, nx> L_diag;
        Eigen::Matrix<double, 1, nu> K_diag;
        if (mpc_weight_preset == "angular_medium") {
          L_diag << 100.0, 100.0, 10.0, 1.0, 200.0, 1.0, 8.0, 8.0, 2.0,
              100.0, 10.0, 1.0;
        } else if (mpc_weight_preset == "angular_mild") {
          L_diag << 80.0, 80.0, 10.0, 1.0, 200.0, 1.0, 4.0, 4.0, 1.0,
              100.0, 10.0, 1.0;
        } else {
          // Previous default:
          // L_diag << 50.0, 50.0, 10.0, 1.0, 200.0, 1.0, 1.0, 1.0,
          //     0.5, 100.0, 10.0, 1.0;
          L_diag << 50.0, 50.0, 80.0, 1.0, 200.0, 1.0, 1.0, 1.0, 10.0,
              100.0, 10.0, 1.0;
        }
        if (has_mpc_l_diag_override) {
          L_diag = mpc_l_diag_override_values;
        }
        K_diag << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0;
        MPC_solv.set_weight(1e-6, L_diag, K_diag);

        Eigen::VectorXd pos_des =
            kinDynSolver.integrateDIY(RobotState.q, RobotState.wbc_delta_q_final);
        RobotState.motors_pos_des =
            eigen2std(pos_des.block(7, 0, model_nv - 6, 1));
        RobotState.motors_vel_des = eigen2std(RobotState.wbc_dq_final);
        RobotState.motors_tor_des = eigen2std(RobotState.wbc_tauJointRes);
      }

      pvtCtr.dataBusRead(RobotState);
      if (simTime <= startSteppingTime) {
        pvtCtr.calMotorsPVT(100.0 / 1000.0 / 180.0 * 3.1415);
      } else {
        pvtCtr.setJointPD(100 * walk_leg_pd_scale,
                           10 * walk_leg_pd_scale, "J_ankle_l_pitch");
        pvtCtr.setJointPD(100 * walk_leg_pd_scale,
                           10 * walk_leg_pd_scale, "J_ankle_l_roll");
        pvtCtr.setJointPD(100 * walk_leg_pd_scale,
                           10 * walk_leg_pd_scale, "J_ankle_r_pitch");
        pvtCtr.setJointPD(100 * walk_leg_pd_scale,
                           10 * walk_leg_pd_scale, "J_ankle_r_roll");
        pvtCtr.setJointPD(1000 * walk_leg_pd_scale,
                           100 * walk_leg_pd_scale, "J_knee_l_pitch");
        pvtCtr.setJointPD(1000 * walk_leg_pd_scale,
                           100 * walk_leg_pd_scale, "J_knee_r_pitch");
        pvtCtr.calMotorsPVT();
      }
      pvtCtr.dataBusWrite(RobotState);

      mj_interface.setMotorsTorque(RobotState.motors_tor_out);

	      RobotState.vel_track_error = computeVelTrackError(RobotState);
	      RobotState.torso_angle_error = computeTorsoAngleError(RobotState);
	      RobotState.tau_bias_norm = RobotState.tau_non_com.norm();
	      const double wbc_delta_fr_norm =
	          RobotState.wbc_FrRes.size() == RobotState.Fr_ff.size()
	              ? (RobotState.wbc_FrRes - RobotState.Fr_ff).norm()
	              : std::numeric_limits<double>::quiet_NaN();
		      RobotState.fall_detected = detectFall(
		          RobotState,
		          fallHeightThreshold + RobotState.stair_support_height,
		          fallAngleThreshold);

      logger.startNewLine();
      logger.recItermData("simTime", simTime);
      logger.recItermData("exp_id", static_cast<double>(exp));
      logger.recItermData("use_vicm",
                          use_variable_inertia_model ? 1.0 : 0.0);
      logger.recItermData("use_tau_bias",
                          use_tau_bias_feedforward ? 1.0 : 0.0);
      logger.recItermData("leg_mass_fraction", leg_mass_fraction);
      logger.recItermData("speed_ref_x", command_speed_x);
      logger.recItermData("speed_ref_y", command_speed_y);
      logger.recItermData("push_force_cmd", RobotState.push_force_cmd);
      logger.recItermData("gait_phase", RobotState.phi);
      logger.recItermData("leg_state", static_cast<double>(RobotState.legState));
      logger.recItermData("step_count", static_cast<double>(RobotState.step_count));
      logger.recItermData("vel_track_error", RobotState.vel_track_error);
      logger.recItermData("torso_angle_error", RobotState.torso_angle_error);
      logger.recItermData("tau_bias_norm", RobotState.tau_bias_norm);
      logger.recItermData("tau_mpc", RobotState.tau_non_mpc);
      logger.recItermData("tau_mpc_norm", RobotState.tau_non_mpc.norm());
      logger.recItermData("tau_idot_omega", RobotState.tau_non_idot_omega);
      logger.recItermData("tau_idot_omega_norm",
                          RobotState.tau_non_idot_omega.norm());
      logger.recItermData("tau_gyro", RobotState.tau_non_gyro);
      logger.recItermData("tau_gyro_norm", RobotState.tau_non_gyro.norm());
      logger.recItermData("mpc_qp_status",
                          static_cast<double>(RobotState.qpStatus_MPC));
      logger.recItermData("mpc_qp_nwsr",
                          static_cast<double>(RobotState.qp_nWSR_MPC));
	      logger.recItermData("wbc_qp_status",
	                          static_cast<double>(RobotState.qp_status));
	      logger.recItermData("wbc_qp_nwsr",
	                          static_cast<double>(RobotState.qp_nWSR));
	      logger.recItermData("wbc_delta_fr_norm", wbc_delta_fr_norm);
	      logger.recItermData("fall_flag", RobotState.fall_detected ? 1.0 : 0.0);
      logger.recItermData("controller_mass", RobotState.controller_mass);
      logger.recItermData("controller_leg_mass", RobotState.controller_leg_mass);
      logger.recItermData("motor_pos_des", RobotState.motors_pos_des);
      logger.recItermData("motor_pos_cur", RobotState.motors_pos_cur);
      logger.recItermData("motor_vel_des", RobotState.motors_vel_des);
      logger.recItermData("motor_vel_cur", RobotState.motors_vel_cur);
      logger.recItermData("motor_tor_des", RobotState.motors_tor_des);
      logger.recItermData("rpyVal", RobotState.rpy);
      logger.recItermData("base_omega_W", RobotState.base_omega_W);
      logger.recItermData("gpsVal", RobotState.basePos);
      logger.recItermData("base_vel", RobotState.dq.block<3, 1>(0, 0));
      logger.recItermData("dX_cal", RobotState.dX_cal);
      logger.recItermData("Ufe", RobotState.Fr_ff);
      logger.recItermData("Xd", RobotState.Xd);
      logger.recItermData("X_cur", RobotState.X_cur);
      logger.recItermData("X_cal", RobotState.X_cal);
      logger.finishLine();

      trace_file << std::fixed << std::setprecision(6) << simTime << ","
                 << static_cast<int>(exp) << ","
                 << (use_variable_inertia_model ? 1 : 0) << ","
                 << (use_tau_bias_feedforward ? 1 : 0) << ","
                 << leg_mass_fraction << "," << command_speed_x << ","
                 << command_speed_y << ","
                 << (pushActive ? 1 : 0) << "," << RobotState.push_force_cmd
                 << "," << RobotState.step_count << "," << RobotState.phi << ","
                 << static_cast<int>(RobotState.legState) << ","
                 << RobotState.q(0) << "," << RobotState.q(1) << ","
                 << RobotState.q(2) << "," << RobotState.base_rpy(0) << ","
                 << RobotState.base_rpy(1) << "," << RobotState.base_rpy(2)
                 << "," << RobotState.js_eul_des(2)
                 << "," << RobotState.dq(0) << "," << RobotState.dq(1) << ","
                 << RobotState.dq(2) << "," << RobotState.dq(3) << ","
                 << RobotState.dq(4) << "," << RobotState.dq(5) << ","
                 << RobotState.js_vel_des(0) << "," << RobotState.js_vel_des(1)
                 << "," << RobotState.js_omega_des(2) << ","
                 << RobotState.vel_track_error << ","
                 << RobotState.torso_angle_error << ","
                 << RobotState.tau_bias_norm << ","
                 << RobotState.tau_non_mpc(0) << ","
                 << RobotState.tau_non_mpc(1) << ","
                 << RobotState.tau_non_mpc(2) << ","
                 << RobotState.tau_non_mpc.norm() << ","
                 << RobotState.tau_non_idot_omega(0) << ","
                 << RobotState.tau_non_idot_omega(1) << ","
                 << RobotState.tau_non_idot_omega(2) << ","
                 << RobotState.tau_non_idot_omega.norm() << ","
                 << RobotState.tau_non_gyro(0) << ","
                 << RobotState.tau_non_gyro(1) << ","
                 << RobotState.tau_non_gyro(2) << ","
                 << RobotState.tau_non_gyro.norm() << ","
	                 << RobotState.qpStatus_MPC << ","
	                 << RobotState.qp_nWSR_MPC << ","
	                 << RobotState.qp_status << ","
	                 << RobotState.qp_nWSR << ","
	                 << wbc_delta_fr_norm << ","
	                 << RobotState.controller_mass << ","
                 << RobotState.controller_leg_mass << ","
                 << RobotState.fL[2] << ","
                 << RobotState.fR[2] << ","
                 << RobotState.foot_contact_fz_raw_l << ","
                 << RobotState.foot_contact_fz_raw_r << ","
                 << RobotState.foot_contact_fz_l << ","
                 << RobotState.foot_contact_fz_r << ","
                 << RobotState.foot_touch_raw_l << ","
                 << RobotState.foot_touch_raw_r << ","
                 << (RobotState.FL_est.size() > 2
                         ? RobotState.FL_est(2)
                         : std::numeric_limits<double>::quiet_NaN())
                 << ","
                 << (RobotState.FR_est.size() > 2
                         ? RobotState.FR_est(2)
                         : std::numeric_limits<double>::quiet_NaN())
                 << ","
                 << (RobotState.fall_detected ? 1 : 0) << ","
                 << (pushPhaseTriggered ? 1 : 0) << ","
                 << pushActualStartTime << "," << recoveryCompletedSteps << ","
                 << static_cast<int>(RobotState.motionState) << ","
                 << (stair_terrain.enabled ? 1 : 0) << ","
                 << stair_terrain.stepHeightAt(RobotState.q(0)) << ","
                 << RobotState.js_pos_des(2) << ","
                 << RobotState.swingDesPosFinal_W(0) << ","
                 << RobotState.swingDesPosFinal_W(1) << ","
                 << RobotState.swingDesPosFinal_W(2) << ","
                 << RobotState.fe_l_pos_W(0) << ","
                 << RobotState.fe_l_pos_W(2) << ","
                 << RobotState.fe_r_pos_W(0) << ","
                 << RobotState.fe_r_pos_W(2) << "\n";

      if (snapshot_enabled && snapshot_index < snapshot_count &&
          simTime + 0.5 * mj_model->opt.timestep >= next_snapshot_time) {
        std::ostringstream path;
        path << snapshot_dir << "/" << snapshot_prefix << "_"
             << std::setw(2) << std::setfill('0') << snapshot_index
             << std::setfill(' ') << "_t" << std::fixed
             << std::setprecision(3) << simTime << ".ppm";
        if (uiController.saveSnapshotPPM(path.str())) {
          std::cout << "[Snapshot] saved " << path.str() << std::endl;
        } else {
          std::cerr << "[Snapshot] failed to save " << path.str()
                    << std::endl;
        }
        ++snapshot_index;
        if (snapshot_times.empty()) {
          next_snapshot_time =
              snapshot_start_time + snapshot_interval * snapshot_index;
        } else if (snapshot_index < static_cast<int>(snapshot_times.size())) {
          next_snapshot_time = snapshot_times[snapshot_index];
        } else {
          next_snapshot_time = std::numeric_limits<double>::infinity();
        }
        if (snapshot_exit_after_capture && snapshot_index >= snapshot_count) {
          break;
        }
      }

      if (RobotState.fall_detected) {
        fallDetected = true;
        fallTime = simTime;
        break;
      }
    }

    if (fallDetected || mj_data->time >= simEndTime ||
        (snapshot_exit_after_capture && snapshot_enabled &&
         snapshot_index >= snapshot_count)) {
      break;
    }

    if (!headless) {
      uiController.updateScene();
    }
  }

  if (print_mpc_timing && mpcTimingTotalSamples > 0) {
    std::cout << std::fixed << std::setprecision(6)
              << "[MPC timing total] samples=" << mpcTimingTotalSamples
              << " avg_wall_ms="
              << (mpcTimingTotalWallMsSum /
                  static_cast<double>(mpcTimingTotalSamples))
              << " max_wall_ms=" << mpcTimingTotalWallMsMax
              << " avg_qp_ms="
              << (mpcTimingTotalQpMsSum /
                  static_cast<double>(mpcTimingTotalSamples))
              << " max_qp_ms=" << mpcTimingTotalQpMsMax << std::endl;
  }

  if (mpc_horizon_file.is_open()) {
    for (const auto &completed : completedMpcPredictions) {
      const PendingMpcPrediction &prediction = completed.prediction;
      const Eigen::Vector3d error =
          completed.actual_omega - prediction.predicted_omega;
      const Eigen::Vector3d actualDelta =
          completed.actual_omega - prediction.start_omega;
      mpc_horizon_file
          << std::fixed << std::setprecision(9) << prediction.origin_time
          << "," << prediction.target_time << "," << completed.actual_time
          << "," << prediction.horizon_steps << "," << controller_label
          << "," << static_cast<int>(exp) << "," << leg_mass_fraction << ","
          << prediction.origin_phi << ","
          << static_cast<int>(prediction.origin_leg_state) << ","
          << prediction.origin_wz_ref << "," << prediction.start_omega(0)
          << "," << prediction.start_omega(1) << ","
          << prediction.start_omega(2) << ","
          << prediction.predicted_omega(0) << ","
          << prediction.predicted_omega(1) << ","
          << prediction.predicted_omega(2) << ","
          << completed.actual_omega(0) << ","
          << completed.actual_omega(1) << ","
          << completed.actual_omega(2) << "," << error(0) << ","
          << error(1) << "," << error(2) << "," << error.norm() << ","
          << actualDelta(0) << "," << actualDelta(1) << ","
          << actualDelta(2) << "," << actualDelta.norm() << "\n";
    }
  }

  summary_file << static_cast<int>(exp) << "," << exp_name << ","
               << (use_variable_inertia_model ? 1 : 0) << ","
               << (use_tau_bias_feedforward ? 1 : 0) << ","
               << leg_mass_fraction << "," << target_speed_x << ","
               << target_speed_y << ","
               << push_force << "," << push_start_time << "," << push_duration
               << "," << stepCount << "," << (fallDetected ? 1 : 0) << ","
               << fallTime << "," << simTime << ","
               << RobotState.controller_mass << ","
               << RobotState.controller_leg_mass << "\n";

  if (render_enabled) {
    uiController.Close();
  } else {
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);
  }
  trace_file.close();
  fr_ff_file.close();
  if (pred_error_file.is_open()) {
    pred_error_file.close();
  }
  if (mpc_horizon_file.is_open()) {
    mpc_horizon_file.close();
  }
  summary_file.close();
  return 0;
}
