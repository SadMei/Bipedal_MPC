/*
This is part of OpenLoong Dynamics Control, an open project for the control of biped robot,
Copyright (C) 2024 Humanoid Robot (Shanghai) Co., Ltd, under Apache 2.0.
Feel free to use in any purpose, and cite OpenLoong-Dynamics-Control in any style, to contribute to the advancement of the community.
 <https://atomgit.com/openloong/openloong-dyn-control.git>
 <web@openloong.org.cn>
*/

#pragma once

#include <Eigen/Dense>
#include <deque> // 新增：用于历史数据缓冲
#include "data_bus.h"

class FootPlacement {
 public:
		double kp_vx{0}, kp_vy{0}, kp_wz{0};
		double legLength{1};
		double stepHeight{0.1};
		double xOff_L{-0.01};
		double yOff_L{0.01};
		double zOff_W{-0.035};
		double lookaheadTime{-1.0};
		double firstStepLateralBiasScale{0.25};
		double firstStepHeightScale{0.6};
		StairTerrainProfile stairTerrain;
		double stairFootContactOffset{0.0};
		double stairLandingMargin{0.10};
		double phi{0};    // phase varialbe for trajectory generation, must between 0 and 1
		double tSwing{0.4}; // swing time

	// --- 仿生控制参数 ---
	double retraction_ratio{0.1};
	double l_thigh{0.4}; //大腿腿长
	double l_shank{0.387}; //小腿腿长
	bool use_bio_height{true}; // 建议开启以测试效果
	double dt{0.001};          // 控制周期，用于计算 buffer 大小
	// -------------------

	Eigen::Vector3d posStart_W, posDes_W, hipPos_W, STPos_W;
	Eigen::Vector3d desV_W, curV_W;
	double desWz_W;
	Eigen::Vector3d base_pos;

	double Trajectory(double phase, double des1, double des2);
	void getSwingPos();
	void dataBusRead(DataBus &robotState);
	void dataBusWrite(DataBus &robotState);
	DataBus::LegState legState;
	uint32_t timer = 0;
	double InverseKinematicsHip(const Eigen::Vector3d& target_pos_rel);
	Eigen::Vector3d ForwardKinematics(double q_hip, double q_knee);
 private:
	// 仿生辅助函数
	double getBioInspiredKnee(double q_hip, double v_norm);
	double getBioZOffset(double phase, double swing_amplitude_x, double velocity);

	// --- 历史数据管理 ---
	std::deque<double> l_hip_pitch_hist;
	std::deque<double> r_hip_pitch_hist;
	int history_size; // 缓冲区大小
	Eigen::VectorXd q_current; // 存储当前关节角度

	// 关节索引 (需与 URDF/WBC 保持一致)
	const int L_HIP_PITCH_IDX = 28;
	const int R_HIP_PITCH_IDX = 34;
	// -------------------

	double pDesCur[3]{0};
	double pDesCur_moni[3]{0};
	double yawCur;
	double theta0;
	double omegaZ_W;
	double hip_width;

		uint8_t printcounter = 0;
	uint32_t stepCount{0};
	bool stairTargetInitialized{false};
	uint32_t stairTargetStepCount{0};
	int stairTargetIndex{0};
	};
