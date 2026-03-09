/*
This is part of OpenLoong Dynamics Control, an open project for the control of biped robot,
Copyright (C) 2024 Humanoid Robot (Shanghai) Co., Ltd, under Apache 2.0.
Feel free to use in any purpose, and cite OpenLoong-Dynamics-Control in any style, to contribute to the advancement of the community.
 <https://atomgit.com/openloong/openloong-dyn-control.git>
 <web@openloong.org.cn>
*/
#include "foot_placement.h"
#include "bezier_1D.h"
#include <cmath>
#include <iostream>
#include "serialport.h"

void FootPlacement::dataBusRead(DataBus &robotState) {
	// ... (原有读取逻辑保持不变) ...
	posStart_W=robotState.swingStartPos_W;
	desV_W=robotState.js_vel_des;
	desWz_W=robotState.js_omega_des(2);
	curV_W=robotState.dq.block<3,1>(0,0);
	phi=robotState.phi;
	hipPos_W=robotState.posHip_W;
	STPos_W=robotState.posST_W;
	base_pos=robotState.base_pos;
	tSwing= robotState.tSwing;
	theta0=robotState.theta0;
	yawCur=robotState.rpy[2];
	omegaZ_W=robotState.base_omega_W(2);
	hip_width=robotState.width_hips;
	legState=robotState.legState;

	// --- 缓冲逻辑 ---
	q_current = robotState.q;
	history_size = static_cast<int>(0.26 / dt);

//	if (q_current.size() > std::max(L_HIP_PITCH_IDX, R_HIP_PITCH_IDX)) {
		l_hip_pitch_hist.push_front(q_current(L_HIP_PITCH_IDX));
		r_hip_pitch_hist.push_front(q_current(R_HIP_PITCH_IDX));
		while (l_hip_pitch_hist.size() > history_size) l_hip_pitch_hist.pop_back();
		while (r_hip_pitch_hist.size() > history_size) r_hip_pitch_hist.pop_back();
//	} else {
//		l_hip_pitch_hist.push_front(0.0);
//		r_hip_pitch_hist.push_front(0.0);
//		if (l_hip_pitch_hist.size() > history_size) l_hip_pitch_hist.pop_back();
//		if (r_hip_pitch_hist.size() > history_size) r_hip_pitch_hist.pop_back();
//	}
}

void FootPlacement::dataBusWrite(DataBus &robotState) {
	robotState.swingDesPosCur_W<<pDesCur[0],pDesCur[1],pDesCur[2];
	robotState.swingDesPosFinal_W=posDes_W;
	robotState.swing_fe_rpy_des_W<<0,0,robotState.base_rpy_des(2);
	robotState.swing_fe_pos_des_W<<pDesCur[0],pDesCur[1],pDesCur[2];
}

double FootPlacement::getBioInspiredKnee(double q_hip, double v_norm) {
	double q_hip_norm = 2 * ((q_hip * 180.0 / 3.1415926) - (-20.0)) / 55.0 - 1.0;
	double q_target = -0.7104 + 0.0124 * v_norm
		- 0.8533 * q_hip_norm
		+ 0.0595 * v_norm * q_hip_norm
		+ 1.7419 * std::pow(q_hip_norm, 2)
		- 0.0622 * v_norm * std::pow(q_hip_norm, 2);
	double q_knee_final = -((q_target + 1.0) * 35.0 - 5.0) * 3.1415926 / 180.0;
	return q_knee_final;
}

void FootPlacement::getSwingPos() {
	// =================================================================
	// 步骤 1: 计算理想 Raibert 落点 (作为初值)
	// =================================================================
	Eigen::Matrix3d KP, Rz;
	KP.setZero();
	KP(0,0)=kp_vx; KP(1,1)=kp_vy; KP(2,2)=0;
	Rz<<cos(yawCur),-sin(yawCur),0, sin(yawCur),cos(yawCur),0, 0,0,1;
	KP=Rz*KP*Rz.transpose();

	// Raibert 理想目标 (Unconstrained)
	Eigen::Vector3d posDes_Ideal = hipPos_W + KP*(desV_W-curV_W)*(-1) + 0.5*tSwing*curV_W + curV_W*(1-phi)*tSwing;

	// 转向修正
	double thetaF = yawCur+theta0+omegaZ_W*(1-phi)*tSwing+0.5*omegaZ_W*tSwing+kp_wz*(omegaZ_W-desWz_W);
	posDes_Ideal(0)+=0.5*hip_width* (cos(thetaF)-cos(yawCur+theta0));
	posDes_Ideal(1)+=0.5*hip_width* (sin(thetaF)-sin(yawCur+theta0));

	// 偏置
	double xOff_L=-0.01; double yOff_L=0.01; double zOff_W=-0.035;
	double xOff_W = (legState==DataBus::LSt) ? (cos(yawCur)*xOff_L - sin(yawCur)*yOff_L) : (cos(yawCur)*xOff_L - sin(yawCur)*(-yOff_L));
	double yOff_W = (legState==DataBus::LSt) ? (sin(yawCur)*xOff_L + cos(yawCur)*yOff_L) : (sin(yawCur)*xOff_L + cos(yawCur)*(-yOff_L));
	posDes_Ideal(0)+= xOff_W;
	posDes_Ideal(1)+= yOff_W;

	// 初始高度设为地面以下 (接触高度)
	posDes_Ideal(2) = base_pos(2) - legLength + zOff_W;

	// =================================================================
	// 步骤 2: 迭代求解耦合问题 (Coupling Solver)
	// =================================================================

	// 定时器判断
	timer++;
	bool use_bio = (timer > 0 && use_bio_height && !l_hip_pitch_hist.empty() && phi < 1.0);
	use_bio = 0;
	if (use_bio) {
		// 迭代初始化
		// 当前时刻的 XY 目标 (基于 Raibert 理想终点插值)
		double s_xy = (phi - sin( 2 * 3.1415 * phi)/( 2 * 3.1415));
		if (phi >= 1.0) s_xy = 1.0;

		Eigen::Vector3d pDes_Iter = posDes_Ideal; // 初始猜测：终点就是 Raibert 点

		// 当前时刻的目标位置 pDesCur (需要迭代更新)
		double pDesCur_X = posStart_W(0) + (pDes_Iter(0) - posStart_W(0)) * s_xy;
		double pDesCur_Y = posStart_W(1) + (pDes_Iter(1) - posStart_W(1)) * s_xy;
		double pDesCur_Z = posStart_W(2); // 初值不重要

		double v_norm = curV_W.norm();

		// A. 获取历史髋关节 (输入 1)
		double q_hip_delayed = (legState == DataBus::LSt) ? r_hip_pitch_hist.back() : l_hip_pitch_hist.back();

		// B. 计算仿生膝关节 (Synergy)
		double q_knee_bio = getBioInspiredKnee(q_hip_delayed, v_norm);

		// C. 计算仿生腿长
		double L_bio_sq = l_thigh * l_thigh + l_shank * l_shank + 2 * l_thigh * l_shank * std::cos(q_knee_bio);
		// 安全钳位，防止膝盖反向弯曲等数学错误
		if(L_bio_sq < 0.01) L_bio_sq = 0.01;
//		double L_bio = std::sqrt(L_bio_sq);

		// D. 计算当前需要的水平距离(基于本次迭代的 XY 目标)
		double dx = pDesCur_X - hipPos_W[0];
		double dy = pDesCur_Y - hipPos_W[1];
		double dist_horz_sq = dx * dx + dy * dy;


		pDesCur_Z = hipPos_W[2] - std::sqrt(L_bio_sq - dist_horz_sq);

		// F. 落地融合 (Touchdown Blending)
		// 无论仿生怎么算，最后必须回归地面

		double z_ground = posDes_Ideal(2);
		double blend_ratio = 0.0;
		if (phi > 0.75) {
			blend_ratio = (phi - 0.75) / 0.25;
		}
		pDesCur_Z = (1.0 - blend_ratio) * pDesCur_Z + blend_ratio * z_ground;

		// 安全钳位
		if (pDesCur_Z < 0.02) pDesCur_Z = 0.02;

		// 赋值
		pDesCur[0] = pDesCur_X;
		pDesCur[1] = pDesCur_Y;
		pDesCur[2] = pDesCur_Z;

		// 最终目标 posDes_W 保持为 Raibert 理想值 (为了让 MPC 知道我们的长期意图)
		posDes_W = posDes_Ideal;

	}
	else
	{
		// 降级模式 (Standard)
		posDes_W = posDes_Ideal;
		double s_xy = (phi - sin(2*3.1415*phi)/(2*3.1415));
		pDesCur[0]=posStart_W(0)+(posDes_W(0)-posStart_W(0)) * s_xy;
		pDesCur[1]=posStart_W(1)+(posDes_W(1)-posStart_W(1)) * s_xy;
		pDesCur[2] = posStart_W(2)+Trajectory(0.2, stepHeight, posDes_W(2)-posStart_W(2));
	}
	printcounter++;
	if(printcounter == 5)
	{
		posDes_W = posDes_Ideal;
		double sxy = (phi - sin(2*3.1415*phi)/(2*3.1415));
		pDesCur_moni[0] = posStart_W(0)+(posDes_W(0)-posStart_W(0)) * sxy;
		pDesCur_moni[1] = posStart_W(1)+(posDes_W(1)-posStart_W(1)) * sxy;
		pDesCur_moni[2] = posStart_W(2)+Trajectory(0.2, stepHeight, posDes_W(2)-posStart_W(2));
		serial1.sendFormattedData("%f,%f,%f,%f,%f,%f,%d\r\n",pDesCur[0] ,pDesCur_moni[0],pDesCur[1],pDesCur_moni[1], pDesCur[2],pDesCur_moni[2],timer);
		printcounter = 0;
	}
}

double FootPlacement::Trajectory(double phase, double hei, double len){
	Bezier_1D Bswpid;
	double para0=5, para1=3;
	for(int i=0; i<para0; i++){Bswpid.P.push_back(0.0);}
	for(int i=0; i<para1; i++){Bswpid.P.push_back(1.0);}

	double output;
	if(phi<phase){
		output=hei*Bswpid.getOut(phi/phase);
	}else{
		double s=Bswpid.getOut((1.4-phi)/(1.4-phase));
		if(s>0){
			output=hei*s +len*(1.0-s);
		}else{
			output=len;
		}
	}
	return output;
}