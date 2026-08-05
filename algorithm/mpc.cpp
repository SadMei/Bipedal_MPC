/*
This is part of OpenLoong Dynamics Control, an open project for the control of
biped robot, Copyright (C) 2024 Humanoid Robot (Shanghai) Co., Ltd, under
Apache 2.0. Feel free to use in any purpose, and cite OpenLoong-Dynamics-Control
in any style, to contribute to the advancement of the community.
 <https://gitee.com/panda_23/openloong-dyn-control.git>
 <web@openloong.org.cn>
*/
#include "mpc.h"
#include "useful_math.h"
#include <algorithm>
#include <cstdlib>
#include <string>

namespace {

double getEnvDouble(const char *name, double defaultValue) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return defaultValue;
  }
  char *end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value) {
    return defaultValue;
  }
  return parsed;
}

bool getEnvBool(const char *name, bool defaultValue) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return defaultValue;
  }
  return std::string(value) != "0" && std::string(value) != "false" &&
         std::string(value) != "FALSE";
}

} // namespace

/**
 * @brief MPC类的构造函数
 * @param dtIn 控制周期/时间步长 (s)
 * 继承自QP类，QP(nu*ch, nc*ch) 初始化了二次规划求解器的大小
 * nu*ch 是决策变量的总数 (每个控制周期13个变量 * ch个周期)
 * nc*ch 是约束的总数 (每个控制周期28个约束 * ch个周期)
 */
MPC::MPC(double dtIn) : QP(nu * ch, nc * ch) {
  nominal_m = 77.35;
  // --- 模型物理参数初始化 ---
  m = 77.35; // 机器人总质量 (kg)
  g = -9.8;  // 重力加速度 (m/s^2)，注意这里是负值
  miu = 0.5; // 地面摩擦系数

  // --- Modified: 移除 m_leg，质心动力学不需要手动估算腿部质量 ---

  // --- 足底几何参数 ---
  // 用于计算ZMP（零力矩点）约束的足底边界尺寸 (m)
  delta_foot[0] = 0.073; // 脚前缘到中心的距离
  delta_foot[1] = 0.125; // 脚后缘到中心的距离
  delta_foot[2] = 0.025; // 脚内侧到中心的距离
  delta_foot[3] = 0.025; // 脚外侧到中心的距离

  // --- 控制输入（足底反作用力）的上下限 ---
  // max/min 分别定义了 Fx, Fy, Fz, Mx, My, Mz 的最大值和最小值
  max[0] = 1000.0;
  max[1] = 1000.0;
  max[2] = -3.0 * m * g; // Fx, Fy, Fz 最大值. Fz用-m*g表示反作用力方向
  max[3] = 20.0;
  max[4] = 80.0;
  max[5] = 100.0; // Mx, My, Mz 最大值

  min[0] = -1000.0;
  min[1] = -1000.0;
  min[2] = 0.0; // Fx, Fy, Fz 最小值. Fz最小为0，表示脚不能拉地面
  min[3] = -20.0;
  min[4] = -80.0;
  min[5] = -100.0; // Mx, My, Mz 最小值

  // --- 单刚体模型矩阵初始化 ---
  // Ac, Bc 是连续时间状态空间模型的矩阵 A 和 B
  // A, B 是离散时间状态空间模型的矩阵 A 和 B
  for (int i = 0; i < (mpc_N); i++) {
    Ac[i].setZero();
    Bc[i].setZero();
    A[i].setZero();
    B[i].setZero();
  }
  Cc.setZero();
  C.setZero();

  // --- QP问题构建所需的大矩阵初始化 ---
  Aqp.setZero();  // 将当前状态映射到未来所有状态的矩阵
  Aqp1.setZero(nx * mpc_N, nx * mpc_N); // 用于构建Bqp的中间矩阵
  Bqp1.setZero(nx * mpc_N, nu * mpc_N); // 用于构建Bqp的中间矩阵
  Bqp.setZero();  // 将控制输入映射到未来所有状态的矩阵
  Cqp1.setZero();
  Cqp.setZero();

  // --- 状态与控制变量初始化 ---
  Ufe.setZero();     // 优化的足底反作用力向量 (包含力和力矩)
  Ufe_pre.setZero(); // 上一时刻的足底反作用力

  Xd.setZero();     // 期望的机器人状态轨迹向量 (未来N个时刻)
  X_cur.setZero();  // 机器人当前状态向量
  X_pred.setZero(); // QP模型在预测时域内给出的状态序列
  X_cal.setZero();  // 通过MPC计算出的下一时刻状态
  dX_cal.setZero(); // 通过MPC计算出的状态变化率

  // --- QP代价函数矩阵初始化 ---
  L.setZero(); // 状态误差权重矩阵 Q
  K.setZero(); // 控制输入权重矩阵 R
  alpha = 0.0; // 控制输入的权重系数
  H.setZero(); // QP问题的Hessian矩阵
  c.setZero(); // QP问题的梯度向量

  // --- QP约束矩阵初始化 ---
  u_low.setZero();
  u_up.setZero(); // 决策变量的上下界
  As.setZero();   // 约束矩阵 A
  bs.setZero();   // 约束向量 b (在这里未使用，上下界通过lbA/ubA传递)

  // --- 几何与坐标系变量初始化 ---
  pCoM.setZero();   // 质心位置
  pf2com.setZero(); // 从脚到质心的向量
  pe.setZero();     // 脚的位置
  R_cur.setZero();  // 当前机身旋转矩阵
  R_w2f.setZero();  // 从世界系到支撑脚坐标系的旋转矩阵
  R_f2w.setZero();  // 从支撑脚坐标系到世界系的旋转矩阵
  Ig.setZero();     // 新增：总质心惯量张量，不再初始化硬编码的 Ic
  Ig_dot.setZero();
  Ig_dot_filtered.setZero();
  Ig_dot_bias.setZero();

  // --- 初始化QP求解器 qpOASES ---
  nominal_Ig << 12.61, 0.0, 0.01, 0.0, 11.15, 0.01, 0.01, 0.01, 2.15;
  Ig = nominal_Ig;
  tau_non.setZero();
  tau_non_affine.setZero();
  h_rel_previous.setZero();
  h_rel_dot_filtered.setZero();
  h_rel_dot_initialized = false;
  h_rel_previous_leg_state = -1;
  h_rel_leg_state_initialized = false;
  Ig_dot_filtered.setZero();
  Ig_dot_bias.setZero();
  use_linear_inertia_prediction = false;
  use_linear_tau_dynamics = false;
  use_discrete_momentum_dynamics = false;
  use_ircmpc_rolling_inertia = false;
  has_inertia_horizon = false;
  for (auto &inertia_node : Ig_horizon) {
    inertia_node = nominal_Ig;
  }
  Ig_dot_filter_initialized = false;
  nominal_Ig_initialized = false;
  qp_Status = 0;
  qp_nWSR = 0;
  qp_cpuTime = 0.0;

  qpOASES::Options option;
  option.printLevel = qpOASES::PL_LOW; // 设置求解器输出信息的级别为低
  QP.setOptions(option);

  dt = dtIn; // 保存控制周期
}

/**
 * @brief 设置MPC代价函数的权重
 * @param u_weight 控制输入项的权重系数 alpha
 * @param L_diag 状态误差权重矩阵L的对角线元素
 * @param K_diag 控制输入权重矩阵K的对角线元素
 * 代价函数形式为: J = (X - Xd)'*L*(X - Xd) + U'*K*U
 */
void MPC::set_weight(double u_weight, Eigen::MatrixXd L_diag,
                     Eigen::MatrixXd K_diag) {
  // 临时变量，用于构建大的对角权重矩阵
  Eigen::MatrixXd L_diag_N = Eigen::MatrixXd::Zero(1, nx * mpc_N);
  Eigen::MatrixXd K_diag_N = Eigen::MatrixXd::Zero(1, nu * ch);

  // 重置权重矩阵
  L = Eigen::MatrixXd::Zero(nx * mpc_N, nx * mpc_N);
  K = Eigen::MatrixXd::Zero(nu * ch, nu * ch);

  alpha = u_weight; // 设置控制输入的权重系数

  // 沿对角线填充L矩阵，为每个预测时刻设置权重
  for (int i = 0; i < mpc_N; i++) {
    L_diag_N.block<1, nx>(0, i * nx) = L_diag;
  }

  // 沿对角线填充K矩阵，为每个控制周期设置权重
  for (int i = 0; i < ch; i++) {
    K_diag_N.block<1, nu>(0, i * nu) = K_diag;
  }

  // 将一维向量的值赋给对角矩阵的对角线
  for (int i = 0; i < nx * mpc_N; i++) {
    L(i, i) = L_diag_N(0, i);
  }
  for (int i = 0; i < nu * ch; i++) {
    K(i, i) = K_diag_N(0, i);
  }

  // --- 权重矩阵旋转 ---
  // 将权重矩阵旋转到当前机身的yaw角方向，使得权重在世界坐标系下生效
  // 这样做可以惩罚世界坐标系下的位置/姿态误差，而不是机身坐标系下的误差
  for (int i = 0; i < mpc_N; i++) {
    L.block<3, 3>(i * nx + 3, i * nx + 3) =
        R_curz[i] * L.block<3, 3>(i * nx + 3, i * nx + 3) *
        R_curz[i].transpose(); // 位置权重
    L.block<3, 3>(i * nx + 6, i * nx + 6) =
        R_curz[i] * L.block<3, 3>(i * nx + 6, i * nx + 6) *
        R_curz[i].transpose(); // 角速度权重
    L.block<3, 3>(i * nx + 9, i * nx + 9) =
        R_curz[i] * L.block<3, 3>(i * nx + 9, i * nx + 9) *
        R_curz[i].transpose(); // 线速度权重
  }
  // 同理，旋转控制输入的权重
  for (int i = 0; i < ch; i++) {
    K.block<3, 3>(i * nu, i * nu) =
        R_curz[i] * K.block<3, 3>(i * nu, i * nu) * R_curz[i].transpose();
    K.block<3, 3>(i * nu + 3, i * nu + 3) =
        R_curz[i] * K.block<3, 3>(i * nu + 3, i * nu + 3) *
        R_curz[i].transpose();
    K.block<3, 3>(i * nu + 6, i * nu + 6) =
        R_curz[i] * K.block<3, 3>(i * nu + 6, i * nu + 6) *
        R_curz[i].transpose();
    K.block<3, 3>(i * nu + 9, i * nu + 9) =
        R_curz[i] * K.block<3, 3>(i * nu + 9, i * nu + 9) *
        R_curz[i].transpose();
  }
}

/**
 * @brief 从主数据总线读取机器人状态和期望轨迹
 * @param Data 主数据总线对象
 */
void MPC::dataBusRead(DataBus &Data) {
  // --- 1. 读取并设置当前状态 X_cur ---
  // 状态向量X_cur: [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz]'
  // (12x1)
  X_cur.block<3, 1>(0, 0) = Data.base_rpy;             // 欧拉角 (姿态)
  X_cur.block<3, 1>(3, 0) = Data.q.block<3, 1>(0, 0);  // 质心位置
  X_cur.block<3, 1>(6, 0) = Data.dq.block<3, 1>(3, 0); // 角速度
  X_cur.block<3, 1>(9, 0) = Data.dq.block<3, 1>(0, 0); // 线速度

  // Lock the SRBM inertia at the initial configuration of the active mass
  // distribution. This preserves the fixed-inertia assumption without adding
  // a static parameter mismatch when the leg inertias are scaled.
  if (!nominal_Ig_initialized && Data.inertia.allFinite() &&
      Data.inertia.determinant() > 1e-9) {
    const Eigen::Matrix3d R_yaw = Rz3(X_cur(2));
    nominal_Ig = R_yaw.transpose() * Data.inertia * R_yaw;
    nominal_Ig = 0.5 * (nominal_Ig + nominal_Ig.transpose());
    nominal_Ig_initialized = true;
    std::cout << "[MPC] calibrated fixed SRBM inertia at initial configuration:\n"
              << nominal_Ig << std::endl;
  }

  // --- 2. 更新期望状态轨迹 Xd ---
  if (EN) { // 如果MPC启用
    // 期望轨迹是一个很长的向量，包含了未来N个时刻的期望状态
    // 采用滚动更新的方式：将t+1时刻的期望值移动到t时刻，并在末尾添加新的期望值
    for (int i = 0; i < (mpc_N - 1); i++)
      Xd.block<nx, 1>(nx * i, 0) = Xd.block<nx, 1>(nx * (i + 1), 0);
    // 在轨迹末尾添加从数据总线读取的最新期望值
    for (int j = 0; j < 3; j++)
      Xd(nx * (mpc_N - 1) + j) = Data.js_eul_des(j); // 期望欧拉角
    for (int j = 0; j < 3; j++)
      Xd(nx * (mpc_N - 1) + 3 + j) = Data.js_pos_des(j); // 期望位置
    for (int j = 0; j < 3; j++)
      Xd(nx * (mpc_N - 1) + 6 + j) = Data.js_omega_des(j); // 期望角速度
    for (int j = 0; j < 3; j++)
      Xd(nx * (mpc_N - 1) + 9 + j) = Data.js_vel_des(j); // 期望线速度
  } else { // 如果MPC未启用，将期望状态设置为当前状态，用于保持静止
    for (int i = 0; i < mpc_N; i++) {
      Xd.block<12, 1>(nx * i, 0) = X_cur;
    }
  }

  // --- 3. 计算坐标变换和几何关系 ---
  R_cur = eul2Rot(X_cur(0), X_cur(1), X_cur(2)); // 根据当前欧拉角计算旋转矩阵
  for (int i = 0; i < mpc_N; i++) {
    R_curz[i] = Rz3(X_cur(2)); // 提取只绕Z轴旋转的矩阵，用于模型简化
  }
  pCoM = X_cur.block<3, 1>(3, 0);         // 当前质心位置
  pe.block<3, 1>(0, 0) = Data.fe_l_pos_W; // 左脚世界坐标
  pe.block<3, 1>(3, 0) = Data.fe_r_pos_W; // 右脚世界坐标

  // 计算从脚到质心的向量
  pf2com.block<3, 1>(0, 0) = pe.block<3, 1>(0, 0) - pCoM;
  pf2com.block<3, 1>(3, 0) = pe.block<3, 1>(3, 0) - pCoM;
  // 计算从脚到期望质心的向量
  pf2comd.block<3, 1>(0, 0) = pe.block<3, 1>(0, 0) - Xd.block<3, 1>(3, 0);
  pf2comd.block<3, 1>(3, 0) = pe.block<3, 1>(3, 0) - Xd.block<3, 1>(3, 0);

  // --- Modified: 移除旧的固定惯量 Ic，使用从 DataBus 读取的实时全身质心惯量
  // --- 之前的代码: Ic << 12.61, 0, 0.37, ...; (移除) 之前的代码: Ig = Ic +
  // I_legs; (移除)

  // 这里的 Data.inertia 是由 pino_kin_dyn.cpp 计算的全身质心转动惯量
  // 它随机器人的构型实时变化，体现了质心动力学特性
  const double controller_mass =
      Data.controller_mass > 1e-6 ? Data.controller_mass : nominal_m;
  m = controller_mass;
  max[2] = -3.0 * m * g;

  const double ig_dot_filter_tau =
      getEnvDouble("ODC_IG_DOT_FILTER_TAU", 0.0); // [s], <=0 disables
  if (Data.use_variable_inertia_model) {
    Ig = Data.inertia;
    const Eigen::Matrix3d Ig_dot_raw = Data.inertia_dot;
    if (ig_dot_filter_tau > 0.0 && Ig_dot_raw.allFinite()) {
      const double alpha = dt / (ig_dot_filter_tau + dt);
      if (!Ig_dot_filter_initialized) {
        Ig_dot_filtered = Ig_dot_raw;
        Ig_dot_filter_initialized = true;
      } else {
        Ig_dot_filtered += alpha * (Ig_dot_raw - Ig_dot_filtered);
      }
      Ig_dot = 0.5 * (Ig_dot_filtered + Ig_dot_filtered.transpose());
    } else {
      Ig_dot = Ig_dot_raw;
      Ig_dot_filtered = Ig_dot_raw;
      Ig_dot_filter_initialized = false;
    }
    inertia_is_world_aligned = true;
  } else {
    Ig = nominal_Ig;
    Ig_dot.setZero();
    Ig_dot_filtered.setZero();
    Ig_dot_filter_initialized = false;
    inertia_is_world_aligned = false;
  }
  use_linear_inertia_prediction =
      Data.use_variable_inertia_model && Data.use_linear_inertia_prediction;
  use_discrete_momentum_dynamics =
      Data.use_variable_inertia_model &&
      Data.use_discrete_momentum_dynamics;
  use_ircmpc_rolling_inertia =
      Data.use_variable_inertia_model &&
      Data.use_ircmpc_rolling_inertia && !use_discrete_momentum_dynamics;
  use_linear_tau_dynamics =
      Data.use_variable_inertia_model && Data.use_linear_tau_dynamics &&
      !use_discrete_momentum_dynamics;
  has_inertia_horizon =
      (use_discrete_momentum_dynamics || use_ircmpc_rolling_inertia) &&
      Data.inertia_horizon.size() >= static_cast<size_t>(mpc_N + 1);
  if (has_inertia_horizon) {
    for (int node = 0; node <= mpc_N; ++node) {
      Ig_horizon[node] = Data.inertia_horizon[static_cast<size_t>(node)];
    }
  }
  Ig_dot_bias.setZero();
  tau_non.setZero();
  tau_non_affine.setZero();
  Data.h_rel_dot_mpc.setZero();
  const double tau_non_norm_limit =
      getEnvDouble("ODC_TAU_NON_NORM_LIMIT", 60.0); // [N*m], <=0 disables
  const bool single_support =
      Data.legState == DataBus::LSt || Data.legState == DataBus::RSt;
  const bool tau_phase_allowed =
      !Data.use_tau_phase_gate ||
      (single_support && Data.phi >= Data.tau_phase_gate_min &&
       Data.phi <= Data.tau_phase_gate_max);
  if (Data.use_tau_bias_feedforward && tau_phase_allowed &&
      use_linear_tau_dynamics) {
    Ig_dot_bias = Data.tau_bias_scale * Ig_dot;
    tau_non = Ig_dot_bias * X_cur.block<3, 1>(6, 0);
    const double tau_norm = tau_non.norm();
    if (tau_non_norm_limit > 0.0 && tau_norm > tau_non_norm_limit) {
      const double scale = tau_non_norm_limit / tau_norm;
      Ig_dot_bias *= scale;
      tau_non *= scale;
    }
  }
  const bool use_h_rel_rate =
      getEnvBool("ODC_USE_HREL_RATE", false) &&
      Data.use_variable_inertia_model && use_linear_tau_dynamics;
  const bool reset_h_rel_on_contact_switch =
      getEnvBool("ODC_HREL_RESET_ON_CONTACT_SWITCH", false);
  if (use_h_rel_rate && Data.h_rel.allFinite()) {
    const int current_leg_state = static_cast<int>(Data.legState);
    const bool contact_state_changed =
        h_rel_leg_state_initialized &&
        current_leg_state != h_rel_previous_leg_state;
    h_rel_previous_leg_state = current_leg_state;
    h_rel_leg_state_initialized = true;

    if (reset_h_rel_on_contact_switch && contact_state_changed) {
      // Do not differentiate across a hybrid contact transition. Keep the
      // pre-transition filtered estimate for this MPC update, while resetting
      // the finite-difference baseline to the first sample of the new mode.
      h_rel_previous = Data.h_rel;
      h_rel_dot_initialized = true;
      if (h_rel_dot_filtered.allFinite()) {
        tau_non_affine = h_rel_dot_filtered;
      }
    } else {
      Eigen::Vector3d h_rel_dot_raw = Eigen::Vector3d::Zero();
      if (h_rel_dot_initialized) {
        h_rel_dot_raw = (Data.h_rel - h_rel_previous) / dt;
      } else {
        h_rel_dot_initialized = true;
      }
      h_rel_previous = Data.h_rel;
      const double h_rel_filter_tau =
          getEnvDouble("ODC_HREL_DOT_FILTER_TAU", 0.02);
      if (h_rel_filter_tau > 0.0) {
        const double alpha = dt / (h_rel_filter_tau + dt);
        h_rel_dot_filtered += alpha * (h_rel_dot_raw - h_rel_dot_filtered);
      } else {
        h_rel_dot_filtered = h_rel_dot_raw;
      }
      if (h_rel_dot_filtered.allFinite()) {
        tau_non_affine = h_rel_dot_filtered;
      }
    }
  } else {
    h_rel_dot_initialized = false;
    h_rel_leg_state_initialized = false;
    h_rel_previous_leg_state = -1;
    h_rel_previous.setZero();
    h_rel_dot_filtered.setZero();
  }
  const double tau_affine_norm = tau_non_affine.norm();
  if (tau_non_norm_limit > 0.0 && tau_affine_norm > tau_non_norm_limit) {
    tau_non_affine *= tau_non_norm_limit / tau_affine_norm;
  }
  if (tau_non_affine.norm() > 0.0) {
    tau_non += tau_non_affine;
  }
  // 更新机身惯量矩阵 Ic
  //	Ig << 12.61,  0    , 0.01
  //		  ,0   ,  11.15, 0.01
  //		  ,0.37,  0.01 , 2.15;

  Eigen::Vector3d tau_non_raw = Eigen::Vector3d::Zero();
  if (Data.use_tau_bias_feedforward && tau_phase_allowed &&
      !use_linear_tau_dynamics) {
    tau_non_raw = Data.tau_bias_scale * Data.tau_non_com;
  }
  if (Data.use_tau_bias_feedforward && !use_linear_tau_dynamics &&
      tau_non_raw.allFinite()) {
    tau_non = tau_non_raw;
    tau_non_affine = tau_non;
    const double tau_norm = tau_non.norm();
    if (tau_non_norm_limit > 0.0 && tau_norm > tau_non_norm_limit) {
      tau_non *= tau_non_norm_limit / tau_norm;
      tau_non_affine = tau_non;
    }
  }
  if (use_discrete_momentum_dynamics) {
    tau_non.setZero();
    tau_non_affine.setZero();
  }
  Data.tau_non_mpc = tau_non;
  Data.h_rel_dot_mpc = tau_non_affine;

  // --- 4. 预测未来支撑状态 ---
  legStateCur = Data.legState;      // 当前支撑状态 (左/右/双支撑)
  legStateNext = Data.legStateNext; // 下一个周期的支撑状态
  // 根据当前步态相位phi和步态周期，推算未来N个时刻的支撑腿状态
  const double t_swing_pred = std::max(Data.tSwing, 1e-6);
  for (int i = 0; i < mpc_N; i++) {
    double aa = i * dt / t_swing_pred;
    double phip = Data.phi + aa;
    if (phip > 1) // 如果预测的相位超过1，说明进入了下一个支撑状态
      legState[i] = legStateNext;
    else
      legState[i] = legStateCur;
  }

  // --- 5. 设置支撑平面坐标系 ---
  // 为了简化摩擦锥约束，将约束转换到与地面（支撑平面）平行的坐标系下进行计算
  Eigen::Matrix<double, 3, 3> R_slop;
  R_slop = eul2Rot(Data.slop(0), Data.slop(1), Data.slop(2)); // 地面坡度的旋转矩阵
  if (legStateCur == DataBus::RSt)
    R_f2w = Data.fe_r_rot_W; // 右脚支撑，使用右脚姿态
  else if (legStateCur == DataBus::LSt)
    R_f2w = Data.fe_l_rot_W; // 左脚支撑，使用左脚姿态
  else
    R_f2w = R_slop;          // 双脚支撑，使用地面坡度
  R_w2f = R_f2w.transpose(); // 计算从世界系到支撑脚系的旋转
}

/**
 * @brief 执行MPC计算
 */
void MPC::cal() {
  if (EN) { // 只有在MPC启用时才执行计算
    Eigen::MatrixXd C_seq =
        Eigen::MatrixXd::Zero(nx * mpc_N, 1); // 存储每一预测步的离散偏置
    const bool predict_Ig =
        use_linear_inertia_prediction && inertia_is_world_aligned;
    auto predictedIg = [&](int node) {
      if (has_inertia_horizon && node >= 0 && node <= mpc_N) {
        return Ig_horizon[node];
      }
      Eigen::Matrix3d Ig_i = Ig;
      if (predict_Ig) {
        Ig_i += static_cast<double>(node) * dt * Ig_dot;
        Ig_i = 0.5 * (Ig_i + Ig_i.transpose());
        if (!Ig_i.allFinite() ||
            Eigen::LLT<Eigen::Matrix3d>(Ig_i).info() != Eigen::Success) {
          Ig_i = Ig;
        }
      }
      return Ig_i;
    };
    const bool use_discrete_inertia_sequence =
        use_discrete_momentum_dynamics &&
        (has_inertia_horizon || predict_Ig);
    // --- 1. 构建离散时间状态空间模型 X(k+1) = A*X(k) + B*U(k) ---
    for (int i = 0; i < mpc_N; i++) {
      Ac[i].setZero();
      // 连续时间模型 Ac
      Ac[i].block<3, 3>(0, 6) = R_curz[i].transpose(); // 角速度 -> 姿态变化
      Ac[i].block<3, 3>(3, 9) =
          Eigen::MatrixXd::Identity(3, 3); // 线速度 -> 位置变化
      if (use_linear_tau_dynamics && Ig_dot_bias.norm() > 0.0) {
        Ac[i].block<3, 3>(6, 6) += -predictedIg(i).inverse() * Ig_dot_bias;
      }
      // 离散化: A = I + dt*Ac
      A[i] = Eigen::MatrixXd::Identity(nx, nx) + dt * Ac[i];
      if (use_discrete_inertia_sequence) {
        // World-frame discrete momentum balance, with h_rel neglected:
        // I_{i+1} * omega_{i+1} = I_i * omega_i + dt * tau_i.
        A[i].block<3, 3>(6, 6) =
            predictedIg(i + 1).inverse() * predictedIg(i);
      }
    }
    for (int i = 0; i < mpc_N; i++) {
      Bc[i].setZero();
      // 在每个预测步长更新从脚到质心的向量
      pf2comi[i] = pf2com;
      Eigen::Matrix3d Ic_W_inv;

      // Pinocchio centroidal inertia Ig is already expressed in the CoM frame
      // aligned with the inertial/world frame. Only the legacy nominal SRBM
      // inertia needs the extra yaw rotation into the world frame here.
      if (inertia_is_world_aligned) {
        Ic_W_inv = predictedIg(i).inverse();
      } else {
        Ic_W_inv = (R_curz[i] * Ig * R_curz[i].transpose()).inverse();
      }

      // 连续时间模型 Bc
      // 力矩对角速度的影响 (牛顿-欧拉方程)
      Bc[i].block<3, 3>(6, 0) =
          Ic_W_inv * CrossProduct_A(pf2comi[i].block<3, 1>(
                         0, 0));          // 左脚反作用力产生的力矩
      Bc[i].block<3, 3>(6, 3) = Ic_W_inv; // 左脚反作用力矩
      Bc[i].block<3, 3>(6, 6) =
          Ic_W_inv * CrossProduct_A(pf2comi[i].block<3, 1>(
                         3, 0));          // 右脚反作用力产生的力矩
      Bc[i].block<3, 3>(6, 9) = Ic_W_inv; // 右脚反作用力矩
      // 力对线速度的影响
      Bc[i].block<3, 3>(9, 0) =
          Eigen::MatrixXd::Identity(3, 3) / m; // 左脚反作用力
      Bc[i].block<3, 3>(9, 6) =
          Eigen::MatrixXd::Identity(3, 3) / m; // 右脚反作用力
      // 此处(nx-1, nu-1)似乎是额外项，可能是模型的一部分或者笔误
      Bc[i]((nx - 1), (nu - 1)) = 1.0 / m;
      // 离散化: B = dt*Bc
      B[i] = dt * Bc[i];
      if (use_discrete_inertia_sequence) {
        const Eigen::Matrix3d next_inertia_inv =
            predictedIg(i + 1).inverse();
        B[i].block<3, 3>(6, 0) =
            dt * next_inertia_inv *
            CrossProduct_A(pf2comi[i].block<3, 1>(0, 0));
        B[i].block<3, 3>(6, 3) = dt * next_inertia_inv;
        B[i].block<3, 3>(6, 6) =
            dt * next_inertia_inv *
            CrossProduct_A(pf2comi[i].block<3, 1>(3, 0));
        B[i].block<3, 3>(6, 9) = dt * next_inertia_inv;
      }

      // 新增：计算连续时间的常数偏置项 C_c（包含非线性前馈）
      // Affine bias term induced by the nonlinear centroidal feedforward.
      Eigen::Matrix<double, nx, 1> Cc_i;
      Cc_i.setZero();
      if (!use_linear_tau_dynamics) {
        Cc_i.block<3, 1>(6, 0) =
            -Ic_W_inv * tau_non; // 角加速度的非线性前馈偏置
      } else if (tau_non_affine.norm() > 0.0) {
        Cc_i.block<3, 1>(6, 0) =
            -Ic_W_inv * tau_non_affine; // 内部相对角动量导数的仿射偏置
      }

      // 离散化常数项 C_d
      C_seq.block<nx, 1>(i * nx, 0) = Cc_i * dt;
    }

    // --- 2. 构建QP问题的预测矩阵 ---
    // Aqp: 将初始状态X_cur映射到未来所有时刻状态的矩阵
    for (int i = 0; i < mpc_N; i++)
      Aqp.block<nx, nx>(i * nx, 0) = Eigen::MatrixXd::Identity(nx, nx);
    for (int i = 0; i < mpc_N; i++)
      for (int j = 0; j < i + 1; j++)
        Aqp.block<nx, nx>(i * nx, 0) = A[j] * Aqp.block<nx, nx>(i * nx, 0);

    // Bqp: 将未来所有控制输入U映射到未来所有时刻状态的矩阵
    for (int i = 0; i < mpc_N; i++)
      for (int j = 0; j < i + 1; j++)
        Aqp1.block<nx, nx>(i * nx, j * nx) = Eigen::MatrixXd::Identity(nx, nx);
    for (int i = 1; i < mpc_N; i++)
      for (int j = 0; j < i; j++)
        for (int k = j + 1; k < (i + 1); k++)
          Aqp1.block<nx, nx>(i * nx, j * nx) =
              A[k] * Aqp1.block<nx, nx>(i * nx, j * nx);

    for (int i = 0; i < mpc_N; i++)
      Bqp1.block<nx, nu>(i * nx, i * nu) = B[i];

    Eigen::MatrixXd Bqp11 = Eigen::MatrixXd::Zero(nu * mpc_N, nu * ch);
    Bqp11.setZero();
    Bqp11.block<nu * ch, nu * ch>(0, 0) =
        Eigen::MatrixXd::Identity(nu * ch, nu * ch);
    for (int i = 0; i < (mpc_N - ch); i++)
      Bqp11.block<nu, nu>(nu * ch + i * nu, nu * (ch - 1)) =
          Eigen::MatrixXd::Identity(nu, nu);

    Eigen::MatrixXd B_tmp = Eigen::MatrixXd::Zero(nx * mpc_N, nu * ch);
    B_tmp = Bqp1 * Bqp11;
    Bqp = Aqp1 * B_tmp;
    Eigen::MatrixXd Cqp = Aqp1 * C_seq; // 将前馈偏置累积传播到整个预测域

    // --- 3. 构建QP代价函数 J = 0.5*U'*H*U + c'*U ---
    Eigen::Matrix<double, nu * ch, 1> delta_U;
    delta_U.setZero();
    // 设置一个期望的力分配，用于正则化项，以避免无解
    for (int i = 0; i < ch; i++) {
      if (legState[i] == DataBus::LSt)      // 左脚支撑
        delta_U(nu * i + 2) = m * g;        // 期望左脚承受全部重力
      else if (legState[i] == DataBus::RSt) // 右脚支撑
        delta_U(nu * i + 8) = m * g;        // 期望右脚承受全部重力
      else {                                // 双脚支撑
        delta_U(nu * i + 2) = 0.5 * m * g;  // 期望双脚平分重力
        delta_U(nu * i + 8) = 0.5 * m * g;
      }
    }

    // Hessian矩阵 H = 2 * (Bqp'*L*Bqp + alpha*K)
    H = 2 * (Bqp.transpose() * L * Bqp + alpha * K) +
        1e-10 * Eigen::MatrixXd::Identity(nu * ch,
                                          nu * ch); // 加一个小的正则项防止H奇异
    // 梯度向量 c = 2 * Bqp'*L*(Aqp*X_cur + Cqp - Xd) + 2*alpha*K*delta_U
    c = 2 * Bqp.transpose() * L * (Aqp * X_cur + Cqp - Xd) +
        2 * alpha * K * delta_U;

    const double wrench_rate_weight =
        std::max(0.0, getEnvDouble("ODC_MPC_WRENCH_RATE_WEIGHT", 0.0));
    if (wrench_rate_weight > 0.0) {
      Eigen::Matrix<double, nu * ch, nu * ch> D_rate;
      Eigen::Matrix<double, nu * ch, 1> d_rate;
      D_rate.setZero();
      d_rate.setZero();

      constexpr int wrench_dim = 12;
      D_rate.block<wrench_dim, wrench_dim>(0, 0).setIdentity();
      d_rate.block<wrench_dim, 1>(0, 0) =
          Ufe_pre.block<wrench_dim, 1>(0, 0);
      for (int i = 1; i < ch; ++i) {
        D_rate.block<wrench_dim, wrench_dim>(i * nu, i * nu).setIdentity();
        D_rate.block<wrench_dim, wrench_dim>(i * nu, (i - 1) * nu) =
            -Eigen::Matrix<double, wrench_dim, wrench_dim>::Identity();
      }

      H += 2.0 * wrench_rate_weight * D_rate.transpose() * D_rate;
      c -= 2.0 * wrench_rate_weight * D_rate.transpose() * d_rate;
    }

    // --- 4. 构建QP约束 lbA <= As*U <= ubA 以及 lu <= U <= uu ---

    // 4.1 摩擦锥约束 (F_horizontal <= miu * F_vertical)
    Eigen::Matrix<double, ncfr_single, 3> Asfr111, Asfr11; // 约束矩阵
    Eigen::Matrix<double, ncfr, nu> Asfr1;
    Eigen::Matrix<double, ncfr * ch, nu * ch> Asfr;
    Asfr111.setZero();
    Asfr1.setZero();
    Asfr.setZero();
    Asfr111 << // 线性化的四面体摩擦锥约束
        -1.0,
        0.0, -1.0 / sqrt(2.0) * miu, 1.0, 0.0, -1.0 / sqrt(2.0) * miu, 0.0,
        -1.0, -1.0 / sqrt(2.0) * miu, 0.0, 1.0, -1.0 / sqrt(2.0) * miu;
    Asfr11 = Asfr111 * R_w2f;                   // 将约束旋转到世界坐标系
    Asfr1.block<ncfr_single, 3>(0, 0) = Asfr11; // 应用于左脚
    Asfr1.block<ncfr_single, 3>(ncfr_single, 6) = Asfr11; // 应用于右脚
    for (int i = 0; i < ch; i++)
      Asfr.block<ncfr, nu>(ncfr * i, i * nu) = Asfr1; // 扩展到整个控制周期

    // 4.2 ZMP(零力矩点)约束，防止脚底翻转
    // 此处代码较为复杂，核心思想是保证压力中心在支撑多边形（脚底）内部
    // 通过限制足底力矩Mx, My的范围实现
    // (略去具体实现细节，其结果是构建了Astxy和Astz矩阵)

    // 4.2.1 X和Y方向力矩约束
    double sign_xy[4]{1.0, -1.0, -1.0, 1.0};
    Eigen::Matrix<double, 3, 1> gxyz[4];
    gxyz[0] << 0.0, 1.0, 0.0;
    gxyz[1] << 0.0, 1.0, 0.0;
    gxyz[2] << 1.0, 0.0, 0.0;
    gxyz[3] << 1.0, 0.0, 0.0;
    Eigen::Matrix<double, 3, 1> r[4];
    Eigen::Matrix<double, 3, 1> p[4];
    Eigen::Matrix<double, ncstxya, 6> Astxy_r[4];
    Eigen::Matrix<double, ncstxy_single, 6> Astxy11;
    Eigen::Matrix<double, ncstxy, nu> Astxy1;
    Eigen::Matrix<double, ncstxy * ch, nu * ch> Astxy;
    Astxy_r[0].setZero();
    Astxy_r[1].setZero();
    Astxy_r[2].setZero();
    Astxy_r[3].setZero();
    Astxy11.setZero();
    Astxy1.setZero();
    Astxy.setZero();
    r[0] << 0.0, 1.0, 0.0;
    r[1] << 0.0, 1.0, 0.0;
    r[2] << 1.0, 0.0, 0.0;
    r[3] << 1.0, 0.0, 0.0;
    p[0] << delta_foot[0], 0.0, 0.0;
    p[1] << -delta_foot[1], 0.0, 0.0;
    p[2] << 0.0, delta_foot[2], 0.0;
    p[3] << 0.0, -delta_foot[3], 0.0;
    for (int i = 0; i < 4; i++) {
      Astxy_r[i].block<1, 3>(0, 0) = sign_xy[i] * gxyz[i].transpose() * R_w2f *
                                     R_f2w * r[i] * (R_f2w * r[i]).transpose() *
                                     CrossProduct_A(R_f2w * p[i]);
      Astxy_r[i].block<1, 3>(0, 3) = sign_xy[i] * gxyz[i].transpose() * R_w2f;
      Astxy11.block<ncstxya, 6>(i * ncstxya, 0) = Astxy_r[i];
    }
    Astxy1.block<ncstxy_single, 6>(0, 0) = Astxy11;
    Astxy1.block<ncstxy_single, 6>(ncstxy_single, 6) = Astxy11;
    for (int i = 0; i < ch; i++)
      Astxy.block<ncstxy, nu>(ncstxy * i, nu * i) = Astxy1;

    // 4.2.2 Z方向力矩约束
    Eigen::Matrix<double, ncstza, 6> Astz_r[4];
    Eigen::Matrix<double, ncstz_single, 6> Astz11;
    Eigen::Matrix<double, ncstz, nu> Astz1;
    Eigen::Matrix<double, ncstz * ch, nu * ch> Astz;
    Astz_r[0].setZero();
    Astz_r[1].setZero();
    Astz_r[2].setZero();
    Astz_r[3].setZero();
    Astz11.setZero();
    Astz1.setZero();
    Astz.setZero();
    for (int i = 0; i < 4; i++) {
      Astz_r[i].block<1, 3>(0, 0) =
          -sqrt(p[i](0) * p[i](0) + p[i](1) * p[i](1) + p[i](2) * p[i](2)) *
          miu * Eigen::Matrix<double, 1, 3>(0.0, 0.0, 1.0) * R_w2f;
      Astz_r[i].block<1, 3>(0, 3) =
          Eigen::Matrix<double, 1, 3>(0.0, 0.0, 1.0) * R_w2f;
      Astz_r[i].block<1, 3>(1, 0) = Astz_r[i].block<1, 3>(0, 0);
      Astz_r[i].block<1, 3>(1, 3) = -1 * Astz_r[i].block<1, 3>(0, 3);
      Astz11.block<ncstza, 6>(i * ncstza, 0) = Astz_r[i];
    }
    Astz1.block<ncstz_single, 6>(0, 0) = Astz11;
    Astz1.block<ncstz_single, 6>(ncstz_single, 6) = Astz11;
    for (int i = 0; i < ch; i++)
      Astz.block<ncstz, nu>(ncstz * i, nu * i) = Astz1;

    // 4.3 合并所有约束矩阵
    As.block<ncfr * ch, nu * ch>(0, 0) = Asfr;
    As.block<ncstxy * ch, nu * ch>(ncfr * ch, 0) = Astxy;
    As.block<ncstz * ch, nu * ch>(ncfr * ch + ncstxy * ch, 0) = Astz;
    bs.setZero();

    // 4.4 设置决策变量U的上下界和约束的上下界，这取决于支撑状态
    Eigen::Matrix<double, nu * ch, 1> Guess_value; // 为求解器提供一个初始猜测值
    Guess_value.setZero();
    for (int i = 0; i < ch; i++) {
      if (legState[i] == DataBus::DSt) {        // 双支撑
        Guess_value(i * nu + 2) = -0.5 * m * g; // 初始猜测：双脚平分重力
        Guess_value(i * nu + 8) = -0.5 * m * g;
        // 设置双脚力和力矩的上下限
        for (int j = 0; j < 6; j++) {
          u_low(i * nu + j) = min[j];
          u_low(i * nu + j + 6) = min[j];
          u_up(i * nu + j) = max[j];
          u_up(i * nu + j + 6) = max[j];
        }
      } else if (legState[i] == DataBus::LSt) { // 左支撑
        Guess_value(i * nu + 2) = -m * g;       // 初始猜测：左脚承担全部重力
        Guess_value(i * nu + 8) = 0.0;
        // 设置左脚上下限，右脚（摆动腿）的力和力矩为0
        for (int j = 0; j < 6; j++) {
          u_low(i * nu + j) = min[j];
          u_low(i * nu + j + nu / 2) = 0.0;
          u_up(i * nu + j) = max[j];
          u_up(i * nu + j + nu / 2) = 0.0;
        }
      } else if (legState[i] == DataBus::RSt) { // 右支撑
        Guess_value(i * nu + 2) = 0.0;
        Guess_value(i * nu + 8) = -m * g; // 初始猜测：右脚承担全部重力
        // 设置右脚上下限，左脚（摆动腿）的力和力矩为0
        for (int j = 0; j < 6; j++) {
          u_low(i * nu + j) = 0.0;
          u_low(i * nu + j + nu / 2) = min[j];
          u_up(i * nu + j) = 0.0;
          u_up(i * nu + j + nu / 2) = max[j];
        }
      }
    }

    // --- 5. 求解QP问题 ---
    qpOASES::returnValue res; // 求解结果
    nWSR = 1000000;           // 最大迭代次数
    cpu_time = dt;            // 允许的最大计算时间

    // 设置约束 As*U <= ubA 的上界
    Eigen::Matrix<double, nc * ch, 1> lbA, ubA, one_ch_1;
    one_ch_1.setOnes();
    lbA = -1e7 * one_ch_1; // 下界设为一个很小的负数（相当于无下界）
    ubA.setZero();         // 摩擦锥和ZMP约束都为 <= 0 的形式

    // 根据支撑状态激活或禁用特定约束
    for (int i = 0; i < ch; i++) {
      if (legState[i] == DataBus::DSt) { // 双支撑，所有约束有效
        ubA.block<ncfr, 1>(ncfr * i, 0).setZero();
        ubA.block<ncstxy, 1>(ncfr * ch + ncstxy * i, 0).setZero();
        ubA.block<ncstz, 1>(ncfr * ch + ncstxy * ch + ncstz * i, 0).setZero();
      } else if (legState[i] == DataBus::LSt) { // 左支撑，只对左脚约束有效
        ubA.block<ncfr_single, 1>(ncfr * i, 0).setZero();
        ubA.block<ncstxy_single, 1>(ncfr * ch + ncstxy * i, 0).setZero();
        ubA.block<ncstz_single, 1>(ncfr * ch + ncstxy * ch + ncstz * i, 0)
            .setZero();
      } else if (legState[i] == DataBus::RSt) { // 右支撑，只对右脚约束有效
        ubA.block<ncfr_single, 1>(ncfr * i + ncfr_single, 0).setZero();
        ubA.block<ncstxy_single, 1>(ncfr * ch + ncstxy * i + ncstxy_single, 0)
            .setZero();
        ubA.block<ncstz_single, 1>(
               ncfr * ch + ncstxy * ch + ncstz * i + ncstz_single, 0)
            .setZero();
      }
    }

    // 将Eigen格式的矩阵和向量转换为qpOASES求解器所需的real_t*格式
    copy_Eigen_to_real_t(qp_H, H, nu * ch, nu * ch);
    copy_Eigen_to_real_t(qp_c, c, nu * ch, 1);
    copy_Eigen_to_real_t(qp_As, As, nc * ch, nu * ch);
    copy_Eigen_to_real_t(qp_lbA, lbA, nc * ch, 1);
    copy_Eigen_to_real_t(qp_ubA, ubA, nc * ch, 1);
    copy_Eigen_to_real_t(qp_lu, u_low, nu * ch, 1);
    copy_Eigen_to_real_t(qp_uu, u_up, nu * ch, 1);
    copy_Eigen_to_real_t(xOpt_iniGuess, Guess_value, nu * ch, 1);

    // 调用求解器
    res = QP.init(qp_H, qp_c, qp_As, qp_lu, qp_uu, qp_lbA, qp_ubA, nWSR,
                  &cpu_time, xOpt_iniGuess);

    // 获取求解状态和结果
    qp_Status = qpOASES::getSimpleStatus(res);
    qp_nWSR = nWSR;
    qp_cpuTime = cpu_time;
    // --- 6. 提取并使用优化结果 ---
    if (res == qpOASES::SUCCESSFUL_RETURN && qp_Status == 0) {
      qpOASES::real_t xOpt[nu * ch];
      QP.getPrimalSolution(xOpt); // 获取最优解
      for (int i = 0; i < nu * ch; i++)
        Ufe(i) = xOpt[i]; // 将结果存入Ufe向量
    } else {
      Eigen::Matrix<double, nu, 1> fallback_u = Ufe_pre;
      if (!fallback_u.allFinite()) {
        fallback_u = Guess_value.block<nu, 1>(0, 0);
      }

      auto clearSwingWrench = [](Eigen::Matrix<double, nu, 1> &u,
                                 int support_state) {
        if (support_state == DataBus::LSt) {
          u.block<6, 1>(6, 0).setZero();
        } else if (support_state == DataBus::RSt) {
          u.block<6, 1>(0, 0).setZero();
        }
      };

      for (int i = 0; i < ch; ++i) {
        Eigen::Matrix<double, nu, 1> u_i = fallback_u;
        clearSwingWrench(u_i, legState[i]);
        for (int j = 0; j < nu; ++j) {
          if (!std::isfinite(u_i(j))) {
            u_i(j) = 0.0;
          }
          const int idx = i * nu + j;
          u_i(j) = std::clamp(u_i(j), u_low(idx), u_up(idx));
        }
        Ufe.block<nu, 1>(i * nu, 0) = u_i;
      }
    }

    // 使用连续时间模型结合常数偏置（包含前馈）计算状态导数
    // 由于非线性偏置已经在 C_seq 中引入，直接加上 Cc 即可获得真实加速度
    Eigen::Matrix<double, nx, 1> Cc_inst;
    Cc_inst.setZero();
    Eigen::Matrix3d Ic_W_inv_c;
    if (inertia_is_world_aligned) {
      Ic_W_inv_c = Ig.inverse();
    } else {
      Ic_W_inv_c = (R_curz[0] * Ig * R_curz[0].transpose()).inverse();
    }
    if (!use_linear_tau_dynamics) {
      Cc_inst.block<3, 1>(6, 0) = -Ic_W_inv_c * tau_non;
    } else {
      Cc_inst.block<3, 1>(6, 0) = -Ic_W_inv_c * tau_non_affine;
    }
    dX_cal = Ac[0] * X_cur + Bc[0] * Ufe.block<nu, 1>(0, 0) + Cc_inst;

    // 对预测进行简单的二阶积分补偿，提高精度
    Eigen::Matrix<double, nx, 1> delta_X;
    delta_X.setZero();
    for (int i = 0; i < 3; i++) {
      delta_X(i) = 0.5 * dX_cal(i + 6) * dt * dt;     // 姿态
      delta_X(i + 3) = 0.5 * dX_cal(i + 9) * dt * dt; // 位置
      delta_X(i + 6) = dX_cal(i + 6) * dt;            // 角速度
      delta_X(i + 9) = dX_cal(i + 9) * dt;            // 线速度
    }
    // 保存QP代价函数所使用的完整预测序列。X_cal保留原有的一步
    // 二阶补偿，仅用于现有下层接口。
    X_pred = Aqp * X_cur + Bqp * Ufe + Cqp;
    X_cal = X_pred.block<nx, 1>(0, 0) + delta_X;

    Ufe_pre = Ufe.block<nu, 1>(0, 0); // 保存当前周期的控制输入，以备后用

    QP.reset(); // 重置QP求解器，为下一次计算做准备
  }
}

/**
 * @brief 将计算结果写回主数据总线
 * @param Data 主数据总线对象
 */
void MPC::dataBusWrite(DataBus &Data) {
  // --- 将MPC内部状态和结果传出 ---
  Data.Xd = Xd;                // 期望轨迹
  Data.X_cur = X_cur;          // 当前状态
  Data.fe_react_tau_cmd = Ufe; // 优化出的足底反作用力指令
  Data.X_cal = X_cal;          // 预测的下一时刻状态
  Data.dX_cal = dX_cal;        // 预测的状态导数

  // --- 将QP求解信息传出，用于调试 ---
  Data.qp_nWSR_MPC = nWSR;
  Data.qp_cpuTime_MPC = cpu_time;
  Data.qpStatus_MPC = qp_Status;

  // --- 将计算结果转换为更底层的控制器所需的指令 ---
  Data.Fr_ff = Ufe.block<12, 1>(0, 0); // 提取前馈力指令

  // --- 计算期望的关节/机身加速度、速度和位置，作为下层控制器的输入 ---
  double k = 5; // 一个增益系数
  // 期望的质心水平加速度
  Data.des_ddq.block<2, 1>(0, 0) << dX_cal(9), dX_cal(10);
  // 期望的机身yaw角加速度 (PD控制)
  Data.des_ddq(5) = k * (Xd(6 + 2) - Data.dq(5));

  // 期望的质心和机身速度
  Data.des_dq.block<3, 1>(0, 0) << Xd(9 + 0), Xd(9 + 1), Xd(9 + 2);
  Data.des_dq.block<2, 1>(3, 0) << 0.0, 0.0;
  Data.des_dq(5) = Xd(6 + 2);

  // 期望的位置/姿态变化量
  Data.des_delta_q.block<2, 1>(0, 0) = Data.des_dq.block<2, 1>(0, 0) * dt;
  Data.des_delta_q(5) = Data.des_dq(5) * dt;

  // 最终的机身姿态和位置期望值
  Data.base_rpy_des << 0.005, 0.00, Xd(2);
  Data.base_pos_des << Xd(3 + 0), Xd(3 + 1), Xd(3 + 2);
}

/** @brief 启用MPC */
void MPC::enable() { EN = true; }
/** @brief 禁用MPC */
void MPC::disable() { EN = false; }

/** @brief 获取MPC启用状态 */
bool MPC::get_ENA() { return EN; }

const Eigen::Matrix3d &MPC::getNominalInertia() const { return nominal_Ig; }

bool MPC::hasCalibratedNominalInertia() const {
  return nominal_Ig_initialized;
}

const Eigen::Matrix<double, nx * mpc_N, 1> &
MPC::getPredictedStateSequence() const {
  return X_pred;
}

/**
 * @brief 工具函数：将Eigen格式的矩阵复制到qpOASES使用的real_t类型数组
 * @param target 目标数组指针 (real_t*)
 * @param source 源矩阵 (Eigen::MatrixXd)
 * @param nRows 行数
 * @param nCols 列数
 * @note Eigen默认是列主序存储，而C/C++数组是行主序，此函数进行了正确的转换。
 */
void MPC::copy_Eigen_to_real_t(qpOASES::real_t *target, Eigen::MatrixXd source,
                               int nRows, int nCols) {
  int count = 0;
  for (int i = 0; i < nRows; i++) {
    for (int j = 0; j < nCols; j++) {
      target[count] = source(i, j);
      count++;
    }
  }
}
