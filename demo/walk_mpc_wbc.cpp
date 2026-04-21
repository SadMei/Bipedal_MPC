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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
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

void writeSummaryHeaderIfNeeded(const std::string &file_path) {
  std::ifstream test_in(file_path);
  const bool need_header = !test_in.good() || test_in.peek() == EOF;
  test_in.close();

  if (!need_header) {
    return;
  }

  std::ofstream out(file_path, std::ios::out);
  out << "exp_id,exp_name,use_variable_inertia,use_tau_bias,lambda_leg_scale,"
         "target_speed_x,push_force,push_start_time,push_duration,stable_steps,"
         "fall_detected,fall_time,final_time,controller_mass,controller_leg_mass\n";
}

void applyMuJoCoLegInertiaScale(mjModel *model, mjData *data,
                                double lambda_leg_scale) {
  static constexpr std::array<const char *, 12> kLegBodyNames = {
      "Link_hip_l_roll",   "Link_hip_l_yaw",   "Link_hip_l_pitch",
      "Link_knee_l_pitch", "Link_ankle_l_pitch", "Link_ankle_l_roll",
      "Link_hip_r_roll",   "Link_hip_r_yaw",   "Link_hip_r_pitch",
      "Link_knee_r_pitch", "Link_ankle_r_pitch", "Link_ankle_r_roll"};

  static bool initialized = false;
  static std::array<int, kLegBodyNames.size()> body_ids{};
  static std::array<mjtNum, kLegBodyNames.size()> nominal_body_mass{};
  static std::array<std::array<mjtNum, 3>, kLegBodyNames.size()>
      nominal_body_inertia{};

  if (!initialized) {
    for (size_t idx = 0; idx < kLegBodyNames.size(); ++idx) {
      const int body_id = mj_name2id(model, mjOBJ_BODY, kLegBodyNames[idx]);
      body_ids[idx] = body_id;
      if (body_id < 0) {
        std::cerr << "MuJoCo body not found for lambda scaling: "
                  << kLegBodyNames[idx] << std::endl;
        continue;
      }

      nominal_body_mass[idx] = model->body_mass[body_id];
      for (int axis = 0; axis < 3; ++axis) {
        nominal_body_inertia[idx][axis] = model->body_inertia[3 * body_id + axis];
      }
    }
    initialized = true;
  }

  for (size_t idx = 0; idx < body_ids.size(); ++idx) {
    const int body_id = body_ids[idx];
    if (body_id < 0) {
      continue;
    }

    model->body_mass[body_id] = nominal_body_mass[idx] * lambda_leg_scale;
    for (int axis = 0; axis < 3; ++axis) {
      model->body_inertia[3 * body_id + axis] =
          nominal_body_inertia[idx][axis] * lambda_leg_scale;
    }
  }

  mj_setConst(model, data);
  mj_resetData(model, data);
}
} // namespace

int main(int argc, char **argv) {
  // Experiment selector. The user manually switches this flag.
  int8_t exp = 1; // exp = 1: lambda sweep
                  // exp = 2: speed sweep
                  // exp = 3: tau_bias ablation
                  // exp = 4: disturbance recovery

  // Shared experiment toggles:
  bool use_variable_inertia_model =
      true; // false -> SRBM, true -> VICM/VIBM
  bool use_tau_bias_feedforward =
      true; // used mainly for exp = 3 (false -> ablation)

  // User-editable experiment parameters:
  double lambda_leg_scale = 1.0; // exp = 1
  double target_speed_x = 1.8;   // exp = 2 / 4
  double push_force = 180.0;     // exp = 4, world-frame push along push_dir_W
  double push_start_time = 6.0;  // exp = 4
  double push_duration = 0.15;   // exp = 4
  Eigen::Vector3d push_dir_W(1.0, 0.0, 0.0);
  std::string exp_name = "exp1_lambda_sweep";

  switch (exp) {
  case 1:
    exp_name = "exp1_lambda_sweep";
    target_speed_x = 1.8;
    push_force = 0.0;
    break;
  case 2:
    exp_name = "exp2_speed_sweep";
    lambda_leg_scale = 1.0;
    push_force = 0.0;
    break;
  case 3:
    exp_name = "exp3_tau_bias_ablation";
    lambda_leg_scale = 1.0;
    target_speed_x = 1.8;
    use_tau_bias_feedforward = false;
    push_force = 0.0;
    break;
  case 4:
    exp_name = "exp4_disturbance_recovery";
    lambda_leg_scale = 1.0;
    target_speed_x = 1.8;
    break;
  default:
    std::cerr << "Unsupported experiment id: " << static_cast<int>(exp)
              << std::endl;
    return 1;
  }

  const std::string exp_tag =
      "../record/exp" + std::to_string(static_cast<int>(exp));
  const std::string summary_path = "../record/exp_summary.csv";

  applyMuJoCoLegInertiaScale(mj_model, mj_data, lambda_leg_scale);

  writeSummaryHeaderIfNeeded(summary_path);
  DataLogger logger(exp_tag + "_datalog.log");
  std::ofstream trace_file(exp_tag + "_trace.csv", std::ios::out);
  std::ofstream summary_file(summary_path, std::ios::app);

  trace_file
      << "time,exp_id,use_variable_inertia,use_tau_bias,lambda_leg_scale,"
         "target_speed_x,push_active,push_force,step_count,gait_phase,leg_state,"
         "base_x,base_y,base_z,roll,pitch,yaw,vx,vy,vz,wx,wy,wz,vx_ref,vy_ref,"
         "wz_ref,vel_track_error,torso_angle_error,tau_bias_norm,"
         "controller_mass,controller_leg_mass,fall_detected\n";

  UIctr uiController(mj_model, mj_data);
  MJ_Interface mj_interface(mj_model, mj_data);
  Pin_KinDyn kinDynSolver("../models/AzureLoong.urdf");
  kinDynSolver.applyLegInertiaScale(lambda_leg_scale);
  DataBus RobotState(kinDynSolver.model_nv);
  WBC_priority WBC_solv(kinDynSolver.model_nv, 18, 22, 0.7,
                        mj_model->opt.timestep);
  MPC MPC_solv(dt_200Hz);
  GaitScheduler gaitScheduler(0.5, mj_model->opt.timestep);
  PVT_Ctr pvtCtr(mj_model->opt.timestep, "../common/joint_ctrl_config.json");
  FootPlacement footPlacement;
  JoyStickInterpreter jsInterp(mj_model->opt.timestep);

  uiController.iniGLFW();
  uiController.enableTracking();
  uiController.createWindow("Demo", false);

  const double stand_legLength = 1.05;
  const double foot_height = 0.07;
  const int model_nv = kinDynSolver.model_nv;
  const double startSteppingTime = 2.0;
  const double startWalkingTime = 3.0;
  const double simEndTime = 30.0;
  const double fallHeightThreshold = 0.55;
  const double fallAngleThreshold = 0.8;
  const int baseBodyId = mj_name2id(mj_model, mjOBJ_BODY, "base_link");

  RobotState.width_hips = 0.209;
  footPlacement.kp_vx = 0.1;
  footPlacement.kp_vy = 0.03;
  footPlacement.kp_wz = 0.03;
  footPlacement.stepHeight = 0.205;
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
  logger.addIterm("lambda_leg", 1);
  logger.addIterm("speed_ref_x", 1);
  logger.addIterm("push_force_cmd", 1);
  logger.addIterm("gait_phase", 1);
  logger.addIterm("leg_state", 1);
  logger.addIterm("step_count", 1);
  logger.addIterm("vel_track_error", 1);
  logger.addIterm("torso_angle_error", 1);
  logger.addIterm("tau_bias_norm", 1);
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

  mjtNum simstart = mj_data->time;
  double simTime = mj_data->time;
  while (!glfwWindowShouldClose(uiController.window)) {
    simstart = mj_data->time;
    while (mj_data->time - simstart < 1.0 / 60.0 && uiController.runSim) {
      const bool pushActive =
          exp == 4 && mj_data->time >= push_start_time &&
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
      RobotState.lambda_leg_scale = lambda_leg_scale;
      RobotState.use_variable_inertia_model = use_variable_inertia_model;
      RobotState.use_tau_bias_feedforward = use_tau_bias_feedforward;
      RobotState.target_speed_x = target_speed_x;
      RobotState.push_force_cmd = pushActive ? push_force : 0.0;
      RobotState.push_start_time = push_start_time;
      RobotState.push_duration = push_duration;

      kinDynSolver.dataBusRead(RobotState);
      kinDynSolver.computeJ_dJ();
      kinDynSolver.computeDyn();
      kinDynSolver.dataBusWrite(RobotState);

      if (simTime > startWalkingTime) {
        jsInterp.setVxDesLPara(target_speed_x, 2.0);
        RobotState.motionState = DataBus::Walk;
      } else {
        jsInterp.setIniPos(RobotState.q(0), RobotState.q(1),
                           RobotState.base_rpy(2));
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
        MPC_solv.dataBusRead(RobotState);
        MPC_solv.cal();
        MPC_solv.dataBusWrite(RobotState);
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
        L_diag << 1.0, 1.0, 1.0, 1.0, 200.0, 1.0, 1e-7, 1e-7, 1e-7, 100.0,
            10.0, 1.0;
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
      logger.recItermData("lambda_leg", lambda_leg_scale);
      logger.recItermData("speed_ref_x", target_speed_x);
      logger.recItermData("push_force_cmd", RobotState.push_force_cmd);
      logger.recItermData("gait_phase", RobotState.phi);
      logger.recItermData("leg_state", static_cast<double>(RobotState.legState));
      logger.recItermData("step_count", static_cast<double>(RobotState.step_count));
      logger.recItermData("vel_track_error", RobotState.vel_track_error);
      logger.recItermData("torso_angle_error", RobotState.torso_angle_error);
      logger.recItermData("tau_bias_norm", RobotState.tau_bias_norm);
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
                 << lambda_leg_scale << "," << target_speed_x << ","
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

    uiController.updateScene();
  }

  summary_file << static_cast<int>(exp) << "," << exp_name << ","
               << (use_variable_inertia_model ? 1 : 0) << ","
               << (use_tau_bias_feedforward ? 1 : 0) << ","
               << lambda_leg_scale << "," << target_speed_x << ","
               << push_force << "," << push_start_time << "," << push_duration
               << "," << stepCount << "," << (fallDetected ? 1 : 0) << ","
               << fallTime << "," << simTime << ","
               << RobotState.controller_mass << ","
               << RobotState.controller_leg_mass << "\n";

  uiController.Close();
  trace_file.close();
  summary_file.close();
  return 0;
}
