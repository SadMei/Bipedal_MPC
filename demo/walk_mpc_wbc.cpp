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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

const double dt = 0.001;
const double dt_200Hz = 0.005;
char error[1000] = "Could not load binary model";
mjModel *mj_model = mj_loadXML("../models/scene.xml", 0, error, 1000);
mjData *mj_data = mj_makeData(mj_model);

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
} // namespace

int main(int argc, char **argv) {
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
  double target_speed_x = 1.5;   // exp = 2 / 4
  double target_speed_y = 0.0;   // exp = 2 / 4
  double push_force = 0.0;     // exp = 1 / 4, world-frame push along push_dir_W
  double push_start_time = 6.0;  // exp = 1 / 4
  double push_duration = 0.15;   // exp = 1 / 4
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
    target_speed_x = 0.0;
    target_speed_y = 0.25;
    use_tau_bias_feedforward = false;
    push_force = 0.0;
    break;
  case 4:
    exp_name = "exp4_disturbance_recovery";
    leg_mass_fraction = nominal_leg_mass_fraction;
    target_speed_x = 0.0;
    target_speed_y = 0.25;
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
  target_speed_x = getEnvDouble("ODC_TARGET_SPEED_X", target_speed_x);
  target_speed_y = getEnvDouble("ODC_TARGET_SPEED_Y", target_speed_y);
  push_force = getEnvDouble("ODC_PUSH_FORCE", push_force);
  push_start_time = getEnvDouble("ODC_PUSH_START_TIME", push_start_time);
  push_duration = getEnvDouble("ODC_PUSH_DURATION", push_duration);
  print_variable_inertia =
      getEnvBool("ODC_PRINT_IG", print_variable_inertia);
  print_fr_ff = getEnvBool("ODC_PRINT_FR_FF", print_fr_ff);
  fr_print_interval =
      getEnvDouble("ODC_FR_PRINT_INTERVAL", fr_print_interval);
  const bool print_mpc_timing = getEnvBool("ODC_PRINT_MPC_TIMING", true);
  const double mpc_timing_print_interval =
      getEnvDouble("ODC_MPC_TIMING_PRINT_INTERVAL", 1.0);
  const bool headless = getEnvBool("ODC_HEADLESS", false);
  const double gait_swing_time = getEnvDouble("ODC_TSWING", 0.45);
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

  const double requested_leg_mass_fraction = leg_mass_fraction;
  leg_mass_fraction = std::clamp(leg_mass_fraction, 0.0, 0.8);
  if (std::abs(leg_mass_fraction - requested_leg_mass_fraction) > 1e-9) {
    std::cerr << "Requested leg_mass_fraction=" << requested_leg_mass_fraction
              << " is outside [0.0, 0.8]. Clamped to " << leg_mass_fraction
              << std::endl;
  }

  const std::string exp_tag =
      "../record/exp" + std::to_string(static_cast<int>(exp));
  const std::string summary_path = "../record/exp_summary.csv";
  const std::string fr_ff_path =
      "../record/fr_ff_exp" + std::to_string(static_cast<int>(exp)) + "_" +
      controller_label + "_lf" + std::to_string(leg_mass_fraction) + ".csv";

  applyMuJoCoLegMassFraction(mj_model, mj_data, leg_mass_fraction);

  writeSummaryHeaderIfNeeded(summary_path);
  DataLogger logger(exp_tag + "_datalog.log");
  std::ofstream trace_file(exp_tag + "_trace.csv", std::ios::out);
  std::ofstream fr_ff_file(fr_ff_path, std::ios::out);
  std::ofstream summary_file(summary_path, std::ios::app);

  trace_file
      << "time,exp_id,use_variable_inertia,use_tau_bias,leg_mass_fraction,"
         "target_speed_x,target_speed_y,push_active,push_force,step_count,gait_phase,leg_state,"
         "base_x,base_y,base_z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz,vx_ref,vy_ref,"
         "wz_ref,vel_track_error,torso_angle_error,tau_bias_norm,"
         "tau_mpc_x,tau_mpc_y,tau_mpc_z,tau_mpc_norm,"
         "tau_idot_x,tau_idot_y,tau_idot_z,tau_idot_norm,"
         "tau_gyro_x,tau_gyro_y,tau_gyro_z,tau_gyro_norm,"
         "mpc_qp_status,mpc_qp_nwsr,wbc_qp_status,wbc_qp_nwsr,"
         "controller_mass,controller_leg_mass,fall_detected\n";

  fr_ff_file
      << "time,controller_label,exp_id,use_variable_inertia,use_tau_bias,"
         "leg_mass_fraction,target_speed_x,target_speed_y,push_active,"
         "push_force,step_count,gait_phase,leg_state,"
         "l_fx,l_fy,l_fz,l_tx,l_ty,l_tz,r_fx,r_fy,r_fz,r_tx,r_ty,r_tz,"
         "base_x,base_y,base_z,roll,pitch,yaw,vx,vy,vz,vel_track_error,"
         "torso_angle_error,fall_detected\n";

  UIctr uiController(mj_model, mj_data);
  MJ_Interface mj_interface(mj_model, mj_data);
  kinDynSolver.applyLegMassFraction(leg_mass_fraction);
  DataBus RobotState(kinDynSolver.model_nv);
  WBC_priority WBC_solv(kinDynSolver.model_nv, 18, 22, 0.7,
                        mj_model->opt.timestep);
  MPC MPC_solv(dt_200Hz);
  GaitScheduler gaitScheduler(gait_swing_time, mj_model->opt.timestep);
  std::cout << "[GaitScheduler] tSwing=" << gait_swing_time << std::endl;
  PVT_Ctr pvtCtr(mj_model->opt.timestep, "../common/joint_ctrl_config.json");
  FootPlacement footPlacement;
  JoyStickInterpreter jsInterp(mj_model->opt.timestep);

  if (!headless) {
    uiController.iniGLFW();
    uiController.enableTracking();
    uiController.createWindow("Demo", false);
  }

  const double stand_legLength = 1.05;
  const double foot_height = 0.07;
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

  RobotState.width_hips = 0.209;
  footPlacement.kp_vx = 0.1;
  footPlacement.kp_vy = 0.03;
  footPlacement.kp_wz = 0.03;
  footPlacement.stepHeight = 0.205;
  footPlacement.firstStepLateralBiasScale = 0.25;
  footPlacement.firstStepHeightScale = 0.6;
  footPlacement.legLength = stand_legLength;

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
  double nextIgPrintTime = 0.0;
  double nextFrPrintTime = 0.0;
  double nextMpcTimingPrintTime = 0.0;
  uint64_t mpcTimingSamples = 0;
  double mpcTimingWallMsSum = 0.0;
  double mpcTimingWallMsMax = 0.0;
  double mpcTimingQpMsSum = 0.0;
  double mpcTimingQpMsMax = 0.0;

  mjtNum simstart = mj_data->time;
  double simTime = mj_data->time;
  while (headless || !glfwWindowShouldClose(uiController.window)) {
    simstart = mj_data->time;
    while (mj_data->time - simstart < 1.0 / 60.0 && uiController.runSim) {
      const bool pushEnabled =
          (exp == 1 || exp == 4) && push_force > 0.0 && push_duration > 0.0;
      const bool pushActive =
          pushEnabled && mj_data->time >= push_start_time &&
          mj_data->time < (push_start_time + push_duration);
      for (int i = 0; i < 6; ++i) {
        mj_data->xfrc_applied[6 * baseBodyId + i] = 0.0;
      }
      if (pushActive) {
        const Eigen::Vector3d push_force_W = push_force * push_dir_W.normalized();
        for (int axis = 0; axis < 3; ++axis) {
          mj_data->xfrc_applied[6 * baseBodyId + axis] = push_force_W(axis);
        }
      }

      mj_step(mj_model, mj_data);
      simTime = mj_data->time;

      mj_interface.updateSensorValues();
      mj_interface.dataBusWrite(RobotState);

      RobotState.exp_id = static_cast<int>(exp);
      RobotState.leg_mass_fraction = leg_mass_fraction;
      RobotState.use_variable_inertia_model = use_variable_inertia_model;
      RobotState.use_tau_bias_feedforward = use_tau_bias_feedforward;
      RobotState.target_speed_x = target_speed_x;
      RobotState.target_speed_y = target_speed_y;
      RobotState.push_force_cmd = pushActive ? push_force : 0.0;
      RobotState.push_start_time = push_start_time;
      RobotState.push_duration = push_duration;

      kinDynSolver.dataBusRead(RobotState);
      kinDynSolver.computeJ_dJ();
      kinDynSolver.computeDyn();
      kinDynSolver.dataBusWrite(RobotState);

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
          jsInterp.setVxDesLPara(target_speed_x, startupSpeedRampDuration);
          jsInterp.setVyDesLPara(target_speed_y, startupSpeedRampDuration);
          jsInterp.setWzDesLPara(0.0, startupSpeedRampDuration);
          walkCommandInitialized = true;
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
      RobotState.js_pos_des(2) = stand_legLength + foot_height;
      jsInterp.dataBusWrite(RobotState);

      if (simTime >= startSteppingTime) {
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
          lastLegState = RobotState.legState;
        }
      }

      RobotState.step_count = stepCount;

      MPC_count = MPC_count + 1;
      if (MPC_count > (dt_200Hz / dt - 1)) {
        const auto mpcWallStart = std::chrono::steady_clock::now();
        MPC_solv.dataBusRead(RobotState);
        MPC_solv.cal();
        MPC_solv.dataBusWrite(RobotState);
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
        const bool fr_fall_detected =
            detectFall(RobotState, fallHeightThreshold, fallAngleThreshold);

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
          L_diag << 50.0, 50.0, 10.0, 1.0, 200.0, 1.0, 1.0, 1.0, 0.5,
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
        pvtCtr.setJointPD(100, 10, "J_ankle_l_pitch");
        pvtCtr.setJointPD(100, 10, "J_ankle_l_roll");
        pvtCtr.setJointPD(100, 10, "J_ankle_r_pitch");
        pvtCtr.setJointPD(100, 10, "J_ankle_r_roll");
        pvtCtr.setJointPD(1000, 100, "J_knee_l_pitch");
        pvtCtr.setJointPD(1000, 100, "J_knee_r_pitch");
        pvtCtr.calMotorsPVT();
      }
      pvtCtr.dataBusWrite(RobotState);

      mj_interface.setMotorsTorque(RobotState.motors_tor_out);

      RobotState.vel_track_error = computeVelTrackError(RobotState);
      RobotState.torso_angle_error = computeTorsoAngleError(RobotState);
      RobotState.tau_bias_norm = RobotState.tau_non_com.norm();
      RobotState.fall_detected =
          detectFall(RobotState, fallHeightThreshold, fallAngleThreshold);

      logger.startNewLine();
      logger.recItermData("simTime", simTime);
      logger.recItermData("exp_id", static_cast<double>(exp));
      logger.recItermData("use_vicm",
                          use_variable_inertia_model ? 1.0 : 0.0);
      logger.recItermData("use_tau_bias",
                          use_tau_bias_feedforward ? 1.0 : 0.0);
      logger.recItermData("leg_mass_fraction", leg_mass_fraction);
      logger.recItermData("speed_ref_x", target_speed_x);
      logger.recItermData("speed_ref_y", target_speed_y);
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
                 << leg_mass_fraction << "," << target_speed_x << ","
                 << target_speed_y << ","
                 << (pushActive ? 1 : 0) << "," << RobotState.push_force_cmd
                 << "," << RobotState.step_count << "," << RobotState.phi << ","
                 << static_cast<int>(RobotState.legState) << ","
                 << RobotState.q(0) << "," << RobotState.q(1) << ","
                 << RobotState.q(2) << "," << RobotState.base_rpy(0) << ","
                 << RobotState.base_rpy(1) << "," << RobotState.base_rpy(2)
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
                 << RobotState.controller_mass << ","
                 << RobotState.controller_leg_mass << ","
                 << (RobotState.fall_detected ? 1 : 0) << "\n";

      if (RobotState.fall_detected) {
        fallDetected = true;
        fallTime = simTime;
        break;
      }
    }

    if (fallDetected || mj_data->time >= simEndTime) {
      break;
    }

    if (!headless) {
      uiController.updateScene();
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

  if (!headless) {
    uiController.Close();
  } else {
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);
  }
  trace_file.close();
  fr_ff_file.close();
  summary_file.close();
  return 0;
}
