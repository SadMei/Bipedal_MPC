/*
This is part of OpenLoong Dynamics Control, an open project for the control of biped robot,
Copyright (C) 2024 Humanoid Robot (Shanghai) Co., Ltd, under Apache 2.0.
Feel free to use in any purpose, and cite OpenLoong-Dynamics-Control in any style, to contribute to the advancement of the community.
 <https://atomgit.com/openloong/openloong-dyn-control.git>
 <web@openloong.org.cn>
*/
//
// Created by boxing on 23-12-29.
//

#include "wbc_priority.h"
#include "iostream"
#include "serialport.h"
uint8_t coordinate = 0; //WBC 运动学关节协同
uint8_t fixedarm = 1; //固定上肢关节
//std::vector<std::string> taskOrder_walk;
// QP_nvIn=18, QP_ncIn=22
WBC_priority::WBC_priority(int model_nv_In, int QP_nvIn, int QP_ncIn, double miu_In, double dt) : QP_prob(QP_nvIn,
	QP_ncIn)
{
	timeStep = dt;
	model_nv = model_nv_In;
	miu = miu_In;
	QP_nc = QP_ncIn;
	QP_nv = QP_nvIn;
	Sf = Eigen::MatrixXd::Zero(6, model_nv);
	Sf.block<6, 6>(0, 0) = Eigen::MatrixXd::Identity(6, 6);
	St_qpV2 = Eigen::MatrixXd::Zero(model_nv, model_nv - 6); // 6 means the dims of floating base
	St_qpV2.block(6, 0, model_nv - 6, model_nv - 6) = Eigen::MatrixXd::Identity(model_nv - 6, model_nv - 6);

	St_qpV1 = Eigen::MatrixXd::Zero(model_nv, 6); // 6 means the dims of delta_b
	St_qpV1.block<6, 6>(0, 0) = Eigen::MatrixXd::Identity(6, 6);

	// defined in body frame
	f_z_low = 10;
	f_z_upp = 1400;

	tau_upp_stand_L << 10, 20, 40;    // foot end contact torque limit for stand state, in body frame
	tau_low_stand_L << -10, -20, -40;

	tau_upp_walk_L << 25, 40, 40; // foot end contact torque limit for walk state, in body frame
	tau_low_walk_L << -25, -40, -40;

	qpOASES::Options options;
	options.setToMPC();
	//options.setToReliable();
	options.printLevel = qpOASES::PL_LOW;
	QP_prob.setOptions(options);

	eigen_xOpt = Eigen::VectorXd::Zero(QP_nv);
	eigen_ddq_Opt = Eigen::VectorXd::Zero(model_nv);
	eigen_fr_Opt = Eigen::VectorXd::Zero(12);
	eigen_tau_Opt = Eigen::VectorXd::Zero(model_nv - 6);

	delta_q_final_kin = Eigen::VectorXd::Zero(model_nv);
	dq_final_kin = Eigen::VectorXd::Zero(model_nv);;
	ddq_final_kin = Eigen::VectorXd::Zero(model_nv);

	base_rpy_cur = Eigen::VectorXd::Zero(3);

	// 初始化历史数据缓冲区
	// 计算需要存储的数据点数量以覆盖0.26秒的延迟
	history_size = 0.26 / timeStep; //初始化时使用0.26s作为运动时滞
	// 用初始值(0.0)填充缓冲区
	for (int i = 0; i < history_size; ++i)
	{
		l_hip_q_hist.push_back(0.0);
		r_hip_q_hist.push_back(0.0);
		l_hip_dq_hist.push_back(0.0);
		r_hip_dq_hist.push_back(0.0);
	}

	//  WBC task defined and order build
	///------------ walk --------------
	kin_tasks_walk.addTask("static_Contact");
	kin_tasks_walk.addTask("Roll_Pitch_Yaw_Pz");
	kin_tasks_walk.addTask("RedundantJoints");
	kin_tasks_walk.addTask("PxPy");
	kin_tasks_walk.addTask("SwingLeg");
	kin_tasks_walk.addTask("HandTrack");
	kin_tasks_walk.addTask("HandTrackJoints");
	kin_tasks_walk.addTask("PosRot");
	if (fixedarm) kin_tasks_walk.addTask("FixedArm");
	if (coordinate) kin_tasks_walk.addTask("KneeThighCoordination");

	std::vector<std::string> taskOrder_walk;
	taskOrder_walk.emplace_back("RedundantJoints");
	taskOrder_walk.emplace_back("static_Contact");
//    taskOrder_walk.emplace_back("Roll_Pitch_Yaw_Pz");
//    taskOrder_walk.emplace_back("PxPy");
	if (fixedarm) taskOrder_walk.emplace_back("FixedArm");
	if (!fixedarm) taskOrder_walk.emplace_back("HandTrackJoints");
	taskOrder_walk.emplace_back("PosRot");
	taskOrder_walk.emplace_back("SwingLeg");


	kin_tasks_walk.buildPriority(taskOrder_walk);

	///-------- stand ------------
	kin_tasks_stand.addTask("static_Contact");
	kin_tasks_stand.addTask("CoMTrack");
	kin_tasks_stand.addTask("HandTrackJoints");
	kin_tasks_stand.addTask("HipRPY");
	kin_tasks_stand.addTask("HeadRP");
	kin_tasks_stand.addTask("Pz");
	kin_tasks_stand.addTask("CoMXY_HipRPY");
	kin_tasks_stand.addTask("Roll_Pitch_Yaw");
	kin_tasks_stand.addTask("fixedWaist");

	std::vector<std::string> taskOrder_stand;

//    taskOrder_stand.emplace_back("fixedWaist");
	taskOrder_stand.emplace_back("static_Contact");
//    taskOrder_stand.emplace_back("CoMTrack");
//    taskOrder_stand.emplace_back("HipRPY");
	taskOrder_stand.emplace_back("CoMXY_HipRPY");
	taskOrder_stand.emplace_back("Pz");
	taskOrder_stand.emplace_back("HandTrackJoints");
	taskOrder_stand.emplace_back("HeadRP");

	kin_tasks_stand.buildPriority(taskOrder_stand);
}

void WBC_priority::dataBusRead(const DataBus& robotState)
{
	// foot-end offset posture
	fe_L_rot_L_off = robotState.fe_L_rot_L_off;
	fe_R_rot_L_off = robotState.fe_R_rot_L_off;

	// deisred values
	base_rpy_des = robotState.base_rpy_des;
	base_rpy_cur << robotState.rpy[0], robotState.rpy[1], robotState.rpy[2];
	base_pos_des = robotState.base_pos_des;
	swing_fe_pos_des_W = robotState.swing_fe_pos_des_W;
	swing_fe_rpy_des_W = robotState.swing_fe_rpy_des_W;
	stance_fe_pos_cur_W = robotState.stance_fe_pos_cur_W;
	stance_fe_rot_cur_W = robotState.stance_fe_rot_cur_W;
	stanceDesPos_W = robotState.stanceDesPos_W;
	hd_l_pos_cur_W = robotState.hd_l_pos_W;
	hd_r_pos_cur_W = robotState.hd_r_pos_W;
	hd_l_rot_cur_W = robotState.hd_l_rot_W;
	hd_r_rot_cur_W = robotState.hd_r_rot_W;
	fe_l_pos_cur_W = robotState.fe_l_pos_W;
	fe_r_pos_cur_W = robotState.fe_r_pos_W;
	fe_l_rot_cur_W = robotState.fe_l_rot_W;
	fe_r_rot_cur_W = robotState.fe_r_rot_W;
	des_ddq = robotState.des_ddq;
	des_dq = robotState.des_dq;
	des_delta_q = robotState.des_delta_q;
	des_q = robotState.des_q;

	// state update
	J_base = robotState.J_base;
	dJ_base = robotState.dJ_base;
	base_rot = robotState.base_rot;
	base_pos = robotState.base_pos;
	hip_link_pos = robotState.hip_link_pos;
	hip_link_rot = robotState.hip_link_rot;
	J_hip_link = robotState.J_hip_link;

	Jfe = Eigen::MatrixXd::Zero(12, model_nv);
	Jfe.block(0, 0, 6, model_nv) = robotState.J_l;
	Jfe.block(6, 0, 6, model_nv) = robotState.J_r;
	dJfe = Eigen::MatrixXd::Zero(12, model_nv);
	dJfe.block(0, 0, 6, model_nv) = robotState.dJ_l;
	dJfe.block(6, 0, 6, model_nv) = robotState.dJ_r;
	J_hd_l = robotState.J_hd_l;
	J_hd_r = robotState.J_hd_r;
	dJ_hd_l = robotState.J_hd_l;
	dJ_hd_r = robotState.J_hd_r;
	Fr_ff = robotState.Fr_ff;
	dyn_M = robotState.dyn_M;
	dyn_M_inv = robotState.dyn_M_inv;
	dyn_Ag = robotState.dyn_Ag;
	dyn_dAg = robotState.dyn_dAg;
	dyn_Non = robotState.dyn_Non;
	dq = robotState.dq;
	q = robotState.q;
	legStateCur = robotState.legState;
	motionStateCur = robotState.motionState;

	if (legStateCur == DataBus::LSt)
	{
		Jc = robotState.J_l;
		dJc = robotState.dJ_l;
		Jsw = robotState.J_r;
		dJsw = robotState.dJ_r;
		fe_pos_sw_W = robotState.fe_r_pos_W;
		fe_rot_sw_W = robotState.fe_r_rot_W;
	}
	else
	{
		Jc = robotState.J_r;
		dJc = robotState.dJ_r;
		Jsw = robotState.J_l;
		dJsw = robotState.dJ_l;
		fe_pos_sw_W = robotState.fe_l_pos_W;
		fe_rot_sw_W = robotState.fe_l_rot_W;
	}

	Jcom = robotState.Jcom_W;
	pCoMCur = robotState.pCoM_W;

	des_q = robotState.des_q;
	curV_W = robotState.dq.block<3, 1>(0, 0); // <-- 添加此行，获取期望速度
}

void WBC_priority::dataBusWrite(DataBus& robotState)
{
	robotState.wbc_ddq_final = eigen_ddq_Opt;
	robotState.wbc_tauJointRes = tauJointRes;
	robotState.wbc_FrRes = eigen_fr_Opt;
	robotState.qp_cpuTime = cpu_time;
	robotState.qp_nWSR = nWSR;
	robotState.qp_status = qpStatus;

	robotState.wbc_delta_q_final = delta_q_final_kin;
	robotState.wbc_dq_final = dq_final_kin;
	robotState.wbc_ddq_final = ddq_final_kin;

	robotState.qp_status = qpStatus;
	robotState.qp_nWSR = nWSR;
	robotState.qp_cpuTime = cpu_time;
}

// QP problem contains joint torque, QP_nv=6+12, QP_nc=22;
void WBC_priority::computeTau()
{
	// constust the QP problem, refer to the md file for more details
	Eigen::MatrixXd eigen_qp_A1 = Eigen::MatrixXd::Zero(6, QP_nv);// 18 means the sum of dims of delta_r and delta_Fr
	eigen_qp_A1.block<6, 6>(0, 0) = Sf * dyn_M * St_qpV1;

	eigen_qp_A1.block<6, 12>(0, 6) = -Sf * Jfe.transpose();

	Eigen::VectorXd eqRes = Eigen::VectorXd::Zero(6);
	eqRes = -Sf * dyn_M * ddq_final_kin - Sf * dyn_Non + Sf * Jfe.transpose() * Fr_ff;

	Eigen::Matrix3d Rfe;
	if (motionStateCur == DataBus::Stand)
	{
		Rfe = fe_l_rot_cur_W;
	}
	else
	{
		Rfe = stance_fe_rot_cur_W;
	}

	Eigen::Matrix<double, 12, 12> Mw2b;
	Mw2b.setZero();
	Mw2b.block(0, 0, 3, 3) = Rfe.transpose();
	Mw2b.block(3, 3, 3, 3) = Rfe.transpose();
	Mw2b.block(6, 6, 3, 3) = Rfe.transpose();
	Mw2b.block(9, 9, 3, 3) = Rfe.transpose();

	Eigen::MatrixXd W = Eigen::MatrixXd::Zero(16, 12);
	W(0, 0) = 1;
	W(0, 2) = sqrt(2) / 2.0 * miu;
	W(1, 0) = -1;
	W(1, 2) = sqrt(2) / 2.0 * miu;
	W(2, 1) = 1;
	W(2, 2) = sqrt(2) / 2.0 * miu;
	W(3, 1) = -1;
	W(3, 2) = sqrt(2) / 2.0 * miu;
	W.block<4, 4>(4, 2) = Eigen::MatrixXd::Identity(4, 4);
	W.block<8, 6>(8, 6) = W.block<8, 6>(0, 0);
	W = W * Mw2b;

	Eigen::VectorXd f_low = Eigen::VectorXd::Zero(16);
	Eigen::VectorXd f_upp = Eigen::VectorXd::Zero(16);
	Eigen::Vector3d tau_upp_fe, tau_low_fe;
	if (motionStateCur == DataBus::Stand)
	{
		tau_upp_fe = tau_upp_stand_L;
		tau_low_fe = tau_low_stand_L;
	}
	else
	{
		tau_upp_fe = tau_upp_walk_L;
		tau_low_fe = tau_low_walk_L;
	}
//    std::cout<<"wbc_computeTau, st_fe_rot"<<std::endl<<stance_fe_rot_cur_W<<std::endl;

	f_upp.block<8, 1>(0, 0) << 1e10, 1e10, 1e10, 1e10,
		f_z_upp, tau_upp_fe(0), tau_upp_fe(1), tau_upp_fe(2);
	f_upp.block<8, 1>(8, 0) = f_upp.block<8, 1>(0, 0);
	f_low.block<8, 1>(0, 0) << 0, 0, 0, 0,
		f_z_low, tau_low_fe(0), tau_low_fe(1), tau_low_fe(2);
	f_low.block<8, 1>(8, 0) = f_low.block<8, 1>(0, 0);

	if (motionStateCur == DataBus::Walk || motionStateCur == DataBus::Walk2Stand)
	{
		if (legStateCur == DataBus::LSt)
		{
			f_upp(12) = 0;
			f_upp(13) = 0;
			f_upp(14) = 0;
			f_upp(15) = 0;

			f_low(12) = 0;
			f_low(13) = 0;
			f_low(14) = 0;
			f_low(15) = 0;

			f_low(8) = -1e-7;
			f_low(9) = -1e-7;
			f_low(10) = -1e-7;
			f_low(11) = -1e-7;
		}
		else if (legStateCur == DataBus::RSt)
		{
			f_upp(4) = 0;
			f_upp(5) = 0;
			f_upp(6) = 0;
			f_upp(7) = 0;

			f_low(4) = 0;
			f_low(5) = 0;
			f_low(6) = 0;
			f_low(7) = 0;

			f_low(0) = -1e-7;
			f_low(1) = -1e-7;
			f_low(2) = -1e-7;
			f_low(3) = -1e-7;
		}
	}

	Eigen::MatrixXd eigen_qp_A2 = Eigen::MatrixXd::Zero(16, 18);
	eigen_qp_A2.block<16, 12>(0, 6) = W;
	Eigen::VectorXd neqRes_low = Eigen::VectorXd::Zero(16);
	Eigen::VectorXd neqRes_upp = Eigen::VectorXd::Zero(16);

	neqRes_low = f_low - W * Fr_ff;
	neqRes_upp = f_upp - W * Fr_ff;

	Eigen::MatrixXd eigen_qp_A_final = Eigen::MatrixXd::Zero(QP_nc, QP_nv);
	eigen_qp_A_final.block<6, 18>(0, 0) = eigen_qp_A1;
	eigen_qp_A_final.block<16, 18>(6, 0) = eigen_qp_A2;

	Eigen::VectorXd eigen_qp_lbA = Eigen::VectorXd::Zero(22);
	Eigen::VectorXd eigen_qp_ubA = Eigen::VectorXd::Zero(22);

	eigen_qp_lbA.block<6, 1>(0, 0) = eqRes;
	eigen_qp_lbA.block<16, 1>(6, 0) = neqRes_low;
	eigen_qp_ubA.block<6, 1>(0, 0) = eqRes;
	eigen_qp_ubA.block<16, 1>(6, 0) = neqRes_upp;

	Eigen::MatrixXd eigen_qp_H = Eigen::MatrixXd::Zero(QP_nv, QP_nv);
	Q2 = Eigen::MatrixXd::Identity(6, 6);
	Q1 = Eigen::MatrixXd::Identity(12, 12);
	eigen_qp_H.block<6, 6>(0, 0) = Q2 * 2.0 * 1e7;
	eigen_qp_H.block<12, 12>(6, 6) = Q1 * 2.0 * 1e1;

	// obj: (1/2)x'Hx+x'g
	// s.t. lbA<=Ax<=ubA
	//       lb<=x<=ub
//    qpOASES::real_t qp_H[QP_nv*QP_nv];
//    qpOASES::real_t qp_A[QP_nc*QP_nv];
//    qpOASES::real_t qp_g[QP_nv];
//    qpOASES::real_t qp_lbA[QP_nc];
//    qpOASES::real_t qp_ubA[QP_nc];
//    qpOASES::real_t xOpt_iniGuess[QP_nv];

	copy_Eigen_to_real_t(qp_H, eigen_qp_H, eigen_qp_H.rows(), eigen_qp_H.cols());
	copy_Eigen_to_real_t(qp_A, eigen_qp_A_final, eigen_qp_A_final.rows(), eigen_qp_A_final.cols());
	copy_Eigen_to_real_t(qp_lbA, eigen_qp_lbA, eigen_qp_lbA.rows(), eigen_qp_lbA.cols());
	copy_Eigen_to_real_t(qp_ubA, eigen_qp_ubA, eigen_qp_ubA.rows(), eigen_qp_ubA.cols());

	qpOASES::returnValue res;
	for (int i = 0; i < QP_nv; i++)
	{
		xOpt_iniGuess[i] = 0;
//        xOpt_iniGuess[i] =eigen_xOpt(i);
		qp_g[i] = 0;
	}
	nWSR = 200;
	cpu_time = timeStep;
//    QP_prob.reset();
	res = QP_prob.init(qp_H, qp_g, qp_A, NULL, NULL, qp_lbA, qp_ubA, nWSR, &cpu_time, xOpt_iniGuess);
	qpStatus = qpOASES::getSimpleStatus(res);
//    if (res==qpOASES::SUCCESSFUL_RETURN)
//        printf("WBC-QP: successful_return\n");
//    else if (res==qpOASES::RET_MAX_NWSR_REACHED)
//        printf("WBC-QP: max_nwsr\n");
//    else if (res==qpOASES::RET_INIT_FAILED)
//        printf("WBC-QP: init_failed\n");

	qpOASES::real_t xOpt[QP_nv];
	QP_prob.getPrimalSolution(xOpt);
	if (res == qpOASES::SUCCESSFUL_RETURN)
		for (int i = 0; i < QP_nv; i++)
			eigen_xOpt(i) = xOpt[i];

	eigen_ddq_Opt = ddq_final_kin;
	eigen_ddq_Opt.block<6, 1>(0, 0) += eigen_xOpt.block<6, 1>(0, 0);
	eigen_fr_Opt = Fr_ff + eigen_xOpt.block<12, 1>(6, 0);

	if (qpStatus != 0)
	{
		Eigen::MatrixXd A_x;
		Eigen::VectorXd xOpt_iniGuess_m(QP_nv, 1);
		for (int i = 0; i < QP_nv; i++)
			xOpt_iniGuess_m(i) = xOpt_iniGuess[i];

	}

	Eigen::VectorXd tauRes;
	tauRes = dyn_M * eigen_ddq_Opt + dyn_Non - Jfe.transpose() * eigen_fr_Opt;

	tauJointRes = tauRes.block(6, 0, model_nv - 6, 1);
//    std::cout<<"qpRes_frOpt"<<std::endl;
//    std::cout<<eigen_fr_Opt.transpose()<<std::endl;

	last_nWSR = nWSR;
	last_cpu_time = cpu_time;
}

void WBC_priority::computeDdq(Pin_KinDyn& pinKinDynIn)
{
	// =====================================================================
	// 1. 在函数开头，更新历史数据缓冲区
	// =====================================================================
	delayed++;
	// 警告：您必须为您的机器人模型验证以下关节索引。
	const int L_HIP_IDX = 28; // 对应 q(28), 左腿臀部俯仰关节
	const int L_KNEE_IDX = 29; // 对应 q(29), 左腿膝盖俯仰关节
	const int R_HIP_IDX = 34; // 对应 q(34), 右腿臀部俯仰关节
	const int R_KNEE_IDX = 35; // 对应 q(35), 右腿膝盖俯仰关节

	// 将当前时刻的数据存入缓冲区头部
	l_hip_q_hist.push_front(q(L_HIP_IDX));
	l_hip_dq_hist.push_front(dq(L_HIP_IDX - 1));
	r_hip_q_hist.push_front(q(R_HIP_IDX));
	r_hip_dq_hist.push_front(dq(R_HIP_IDX - 1));
//	printf("%f,%f,%f,%f\r\n",q(L_HIP_IDX),q(R_HIP_IDX),q(L_KNEE_IDX),q(R_KNEE_IDX));

	// 移除超出缓冲区大小的旧数据
	if (l_hip_q_hist.size() > history_size) l_hip_q_hist.pop_back();
	if (l_hip_dq_hist.size() > history_size) l_hip_dq_hist.pop_back();
	if (r_hip_q_hist.size() > history_size) r_hip_q_hist.pop_back();
	if (r_hip_dq_hist.size() > history_size) r_hip_dq_hist.pop_back();

	// task definition
	/// -------- walk -------------
	{
		int id = kin_tasks_walk.getId("static_Contact");
		kin_tasks_walk.taskLib[id].errX = Eigen::VectorXd::Zero(3);
		kin_tasks_walk.taskLib[id].derrX = Eigen::VectorXd::Zero(3);
		kin_tasks_walk.taskLib[id].ddxDes = Eigen::VectorXd::Zero(3);
		kin_tasks_walk.taskLib[id].dxDes = Eigen::VectorXd::Zero(3);
		kin_tasks_walk.taskLib[id].kp = Eigen::MatrixXd::Identity(3, 3) * 0;
		kin_tasks_walk.taskLib[id].kd = Eigen::MatrixXd::Identity(3, 3) * 0;
		kin_tasks_walk.taskLib[id].J = Jc.block(0, 0, 3, model_nv);
//            kin_tasks_walk.taskLib[id].J.b				lock(0,22,3,3).setZero();
		kin_tasks_walk.taskLib[id].dJ = dJc.block(0, 0, 3, model_nv);
//            kin_tasks_walk.taskLib[id].dJ.block(0,22,3,3).setZero();
		kin_tasks_walk.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		id = kin_tasks_walk.getId("RedundantJoints");
		kin_tasks_walk.taskLib[id].errX = Eigen::VectorXd::Zero(5);
		kin_tasks_walk.taskLib[id].errX(0) = 0 - q(21);
		kin_tasks_walk.taskLib[id].errX(1) = 0 - q(22);
		kin_tasks_walk.taskLib[id].errX(2) = 0 - q(23);
		kin_tasks_walk.taskLib[id].errX(3) = 0 - q(24);
		kin_tasks_walk.taskLib[id].errX(4) = 0 - q(25);
		kin_tasks_walk.taskLib[id].derrX = Eigen::VectorXd::Zero(5);
		kin_tasks_walk.taskLib[id].ddxDes = Eigen::VectorXd::Zero(5);
		kin_tasks_walk.taskLib[id].dxDes = Eigen::VectorXd::Zero(5);
		kin_tasks_walk.taskLib[id].kp = Eigen::MatrixXd::Identity(5, 5) * 200;
		kin_tasks_walk.taskLib[id].kd = Eigen::MatrixXd::Identity(5, 5) * 20;
		kin_tasks_walk.taskLib[id].J = Eigen::MatrixXd::Zero(5, model_nv);
		kin_tasks_walk.taskLib[id].J(0, 20) = 1;
		kin_tasks_walk.taskLib[id].J(1, 21) = 1;
		kin_tasks_walk.taskLib[id].J(2, 22) = 1;
		kin_tasks_walk.taskLib[id].J(3, 23) = 1;
		kin_tasks_walk.taskLib[id].J(4, 24) = 1;
		kin_tasks_walk.taskLib[id].dJ = Eigen::MatrixXd::Zero(5, model_nv);
		kin_tasks_walk.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		id = kin_tasks_walk.getId("Roll_Pitch_Yaw_Pz");
		kin_tasks_walk.taskLib[id].errX = Eigen::VectorXd::Zero(4);
		Eigen::Matrix3d desRot = eul2Rot(base_rpy_des(0), base_rpy_des(1), base_rpy_des(2));
		kin_tasks_walk.taskLib[id].errX.block<3, 1>(0, 0) = diffRot(base_rot, desRot);
		kin_tasks_walk.taskLib[id].errX(3) = base_pos_des(2) - q(2);
		kin_tasks_walk.taskLib[id].derrX = Eigen::VectorXd::Zero(4);
		kin_tasks_walk.taskLib[id].derrX.block<3, 1>(0, 0) = -dq.block<3, 1>(3, 0);
		kin_tasks_walk.taskLib[id].derrX(3) = 0 - dq(2);
		kin_tasks_walk.taskLib[id].ddxDes = Eigen::VectorXd::Zero(4);
		kin_tasks_walk.taskLib[id].dxDes = Eigen::VectorXd::Zero(4);
		kin_tasks_walk.taskLib[id].kp = Eigen::MatrixXd::Identity(4, 4) * 2000;
		kin_tasks_walk.taskLib[id].kd = Eigen::MatrixXd::Identity(4, 4) * 100;
		Eigen::MatrixXd taskMap = Eigen::MatrixXd::Zero(4, 6);
		taskMap(0, 3) = 1;
		taskMap(1, 4) = 1;
		taskMap(2, 5) = 1;
		taskMap(3, 2) = 1;
		kin_tasks_walk.taskLib[id].J = taskMap * J_base;
		kin_tasks_walk.taskLib[id].dJ = taskMap * dJ_base;
		kin_tasks_walk.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		id = kin_tasks_walk.getId("PxPy");
		kin_tasks_walk.taskLib[id].errX = Eigen::VectorXd::Zero(2);
		kin_tasks_walk.taskLib[id].errX = des_dq.block(0, 0, 2, 1) * timeStep;
		kin_tasks_walk.taskLib[id].derrX = Eigen::VectorXd::Zero(2);
		kin_tasks_walk.taskLib[id].ddxDes = Eigen::VectorXd::Zero(2);
		kin_tasks_walk.taskLib[id].dxDes = Eigen::VectorXd::Zero(2);
		kin_tasks_walk.taskLib[id].kp = Eigen::MatrixXd::Identity(2, 2) * 500; //100
		kin_tasks_walk.taskLib[id].kd = Eigen::MatrixXd::Identity(2, 2) * 50;
		taskMap = Eigen::MatrixXd::Zero(2, 6);
		taskMap(0, 0) = 1;
		taskMap(1, 1) = 1;
		kin_tasks_walk.taskLib[id].J = taskMap * J_base;
		kin_tasks_walk.taskLib[id].dJ = taskMap * dJ_base;
		kin_tasks_walk.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		id = kin_tasks_walk.getId("PosRot");
		kin_tasks_walk.taskLib[id].errX = Eigen::VectorXd::Zero(6);
		kin_tasks_walk.taskLib[id].errX.block(0, 0, 3, 1) = base_pos_des - q.block(0, 0, 3, 1);
		if (fabs(kin_tasks_walk.taskLib[id].errX(0)) >= 0.02)
			kin_tasks_walk.taskLib			[id].errX(0) = 0.02 * sign(kin_tasks_walk.taskLib[id].errX(0));
		if (fabs(kin_tasks_walk.taskLib[id].errX(1)) >= 0.01)
			kin_tasks_walk.taskLib[id].errX(1) = 0.01 * sign(kin_tasks_walk.taskLib[id].errX(1));
		desRot = eul2Rot(base_rpy_des(0), base_rpy_des(1), base_rpy_des(2));
		kin_tasks_walk.taskLib[id].errX.block<3, 1>(3, 0) = diffRot(base_rot, desRot);
		kin_tasks_walk.taskLib[id].derrX = Eigen::VectorXd::Zero(6);
		kin_tasks_walk.taskLib[id].ddxDes = Eigen::VectorXd::Zero(6);
		kin_tasks_walk.taskLib[id].dxDes = Eigen::VectorXd::Zero(6);
		kin_tasks_walk.taskLib[id].kp = Eigen::MatrixXd::Identity(6, 6) * 10;
		kin_tasks_walk.taskLib[id].kp.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 2000;
		kin_tasks_walk.taskLib[id].kd = Eigen::MatrixXd::Identity(6, 6) * 2;
		kin_tasks_walk.taskLib[id].kd.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 100;
		kin_tasks_walk.taskLib[id].J = J_base;
		kin_tasks_walk.taskLib[id].dJ = dJ_base;
		kin_tasks_walk.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		id = kin_tasks_walk.getId("SwingLeg");
		kin_tasks_walk.taskLib[id].errX = Eigen::VectorXd::Zero(6);
		kin_tasks_walk.taskLib[id].errX.block<3, 1>(0, 0) = swing_fe_pos_des_W - fe_pos_sw_W;
		desRot = eul2Rot(swing_fe_rpy_des_W(0), swing_fe_rpy_des_W(1), swing_fe_rpy_des_W(2));
		kin_tasks_walk.taskLib[id].errX.block<3, 1>(3, 0) = diffRot(fe_rot_sw_W, desRot);
		kin_tasks_walk.taskLib[id].derrX = Eigen::VectorXd::Zero(6);
//        kin_tasks_walk.taskLib[id].derrX=-Jsw*dq;
		kin_tasks_walk.taskLib[id].ddxDes = Eigen::VectorXd::Zero(6);
		kin_tasks_walk.taskLib[id].dxDes = Eigen::VectorXd::Zero(6);

		kin_tasks_walk.taskLib[id].kp = Eigen::MatrixXd::Identity(6, 6) * 2000;
//		kin_tasks_walk.taskLib[id].kp.block<1, 1>(2, 2) = kin_tasks_walk.taskLib[id].kp.block<1, 1>(2, 2) * 0.01;
//		kin_tasks_walk.taskLib[id].kp.block<1, 1>(4, 4) = kin_tasks_walk.taskLib[id].kp.block<1, 1>(4, 4) * 0.1;

		kin_tasks_walk.taskLib[id].kd = Eigen::MatrixXd::Identity(6, 6) * 20;
		kin_tasks_walk.taskLib[id].J = Jsw;
		kin_tasks_walk.taskLib[id].J.block(0, 22, 6, 3).setZero(); // exculde waist joints
		kin_tasks_walk.taskLib[id].dJ = dJsw;
		kin_tasks_walk.taskLib[id].dJ.block(0, 22, 6, 3).setZero(); // exculde waist joints
		kin_tasks_walk.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		if (coordinate && delayed >= 10000)
		{
			// 在 wbc_priority::computeDdq 中
			// 找到 "SwingLeg" 任务的定义部分
			id = kin_tasks_walk.getId("SwingLeg");
			auto& task_swing = kin_tasks_walk.taskLib[id];

			// --- 将其修改为5D (x, y, z, roll, yaw) ---
			task_swing.errX = Eigen::VectorXd::Zero(5);
			task_swing.derrX = Eigen::VectorXd::Zero(5);
			task_swing.ddxDes = Eigen::VectorXd::Zero(5);
			task_swing.dxDes = Eigen::VectorXd::Zero(5);

			// --- 误差计算：选择 x,y,z, roll, yaw 对应的误差 ---
			// 位置误差 (x, y, z)
			task_swing.errX.block<3, 1>(0, 0) = swing_fe_pos_des_W - fe_pos_sw_W;

			// 计算完整的3D姿态误差向量 (roll, pitch, yaw)
			Eigen::Matrix3d desRot = eul2Rot(swing_fe_rpy_des_W(0), swing_fe_rpy_des_W(1), swing_fe_rpy_des_W(2));
			Eigen::Vector3d rot_error = diffRot(fe_rot_sw_W, desRot);

			// 从3D姿态误差中选择 roll 和 yaw
			task_swing.errX(3) = rot_error(0); // roll 误差
			task_swing.errX(4) = rot_error(2); // yaw 误差

			// --- 雅可比矩阵也要选择对应的行 ---
			Eigen::MatrixXd Jsw_full = Jsw;
			task_swing.J = Eigen::MatrixXd::Zero(5, model_nv);
			// 位置部分的雅可比 (x, y, z -> 第0, 1, 2行)
			task_swing.J.block(0, 0, 3, model_nv) = Jsw_full.block(0, 0, 3, model_nv);
			// 姿态部分的雅可比 (roll -> 第3行, yaw -> 第5行)
			task_swing.J.row(3) = Jsw_full.row(3); // roll
			task_swing.J.row(4) = Jsw_full.row(5); // yaw

			// --- dJ也做同样处理 ---
			Eigen::MatrixXd dJsw_full = dJsw;
			task_swing.dJ = Eigen::MatrixXd::Zero(5, model_nv);
			// 位置部分的dJ
			task_swing.dJ.block(0, 0, 3, model_nv) = dJsw_full.block(0, 0, 3, model_nv);
			// 姿态部分的dJ
			task_swing.dJ.row(3) = dJsw_full.row(3); // roll
			task_swing.dJ.row(4) = dJsw_full.row(5); // yaw


			// --- 增益矩阵也要是5x5 ---
			task_swing.kp = Eigen::MatrixXd::Identity(5, 5) * 2000;
			// 对于yaw的控制可以给稍小的增益
			task_swing.kp(4,4) = 1000;

			task_swing.kd = Eigen::MatrixXd::Identity(5, 5) * 100;
			task_swing.kd(4,4) = 60;
		}


		{
			// task 6: hand track
			// define swing arm motion
//			hd_l_pos_L_des<<-0.02, 0.32, -0.159;
//			hd_r_pos_L_des<<-0.02, -0.32, -0.159;
//			hd_l_eul_L_des<<-1.7581, 0.2129, 2.9581;
//			hd_r_eul_L_des<<1.7581, 0.21291, -2.9581;

			Eigen::Vector3d hd_l_eul_L_des = { -1.253, 0.122, -1.732 };
			Eigen::Vector3d hd_r_eul_L_des = { 1.253, 0.122, 1.732 };
			Eigen::Matrix3d hd_l_rot_des = eul2Rot(hd_l_eul_L_des(0), hd_l_eul_L_des(1), hd_l_eul_L_des(2));
			Eigen::Matrix3d hd_r_rot_des = eul2Rot(hd_r_eul_L_des(0), hd_r_eul_L_des(1), hd_r_eul_L_des(2));

			Eigen::Vector3d base2shoulder_l_pos_L_des = { 0.0040, 0.1616, 0.3922 };
			Eigen::Vector3d shoulder2hand_l_pos_L_des = { -0.0240, 0.1584, -0.5512 };
			Eigen::Vector3d base2shoulder_r_pos_L_des = { 0.0040, -0.1616, 0.3922 };
			Eigen::Vector3d shoulder2hand_r_pos_L_des = { -0.0240, -0.1584, -0.5512 };
			double l_hip_pitch = q(28) - qIniDes(28);
			double r_hip_pitch = q(34) - qIniDes(34);
			double k = 0.8;
			hd_l_rot_des = eul2Rot(0, -k * r_hip_pitch, 0) * hd_l_rot_des;
			hd_r_rot_des = eul2Rot(0, -k * l_hip_pitch, 0) * hd_r_rot_des;

			Eigen::Vector3d hd_l_pos_W_des =
				eul2Rot(0, -k * r_hip_pitch, 0) * shoulder2hand_l_pos_L_des + base2shoulder_l_pos_L_des + base_pos;
			Eigen::Vector3d hd_r_pos_W_des =
				eul2Rot(0, -k * l_hip_pitch, 0) * shoulder2hand_r_pos_L_des + base2shoulder_r_pos_L_des + base_pos;

			Eigen::Vector3d hd_l_pos_L_des =
				eul2Rot(0, -k * r_hip_pitch, 0) * shoulder2hand_l_pos_L_des + base2shoulder_l_pos_L_des;
			Eigen::Vector3d hd_r_pos_L_des =
				eul2Rot(0, -k * l_hip_pitch, 0) * shoulder2hand_r_pos_L_des + base2shoulder_r_pos_L_des;

			Eigen::Matrix3d hd_l_rot_W_des = hd_l_rot_des;
			Eigen::Matrix3d hd_r_rot_W_des = hd_r_rot_des;

			id = kin_tasks_walk.getId("HandTrack");
			kin_tasks_walk.taskLib[id].errX = Eigen::VectorXd::Zero(12);
			kin_tasks_walk.taskLib[id].errX.block<3, 1>(0, 0) = hd_l_pos_W_des - hd_l_pos_cur_W;
			kin_tasks_walk.taskLib[id].errX.block<3, 1>(3, 0) = diffRot(hd_l_rot_cur_W, hd_l_rot_W_des);
			kin_tasks_walk.taskLib[id].errX.block<3, 1>(6, 0) = hd_r_pos_W_des - hd_r_pos_cur_W;
			kin_tasks_walk.taskLib[id].errX.block<3, 1>(9, 0) = diffRot(hd_r_rot_cur_W, hd_r_rot_W_des);
			kin_tasks_walk.taskLib[id].derrX = Eigen::VectorXd::Zero(12);
			kin_tasks_walk.taskLib[id].ddxDes = Eigen::VectorXd::Zero(12);
			kin_tasks_walk.taskLib[id].dxDes = Eigen::VectorXd::Zero(12);
			kin_tasks_walk.taskLib[id].kp = Eigen::MatrixXd::Identity(12, 12) * 2000;
			kin_tasks_walk.taskLib[id].kd = Eigen::MatrixXd::Identity(12, 12) * 20;
			kin_tasks_walk.taskLib[id].J = Eigen::MatrixXd::Zero(12, model_nv);
			kin_tasks_walk.taskLib[id].J.block(0, 0, 6, model_nv) = J_hd_l;
			kin_tasks_walk.taskLib[id].J.block(6, 0, 6, model_nv) = J_hd_r;
			kin_tasks_walk.taskLib[id].dJ = Eigen::MatrixXd::Zero(12, model_nv);
			kin_tasks_walk.taskLib[id].dJ.block(0, 0, 6, model_nv) = dJ_hd_l;
			kin_tasks_walk.taskLib[id].dJ.block(6, 0, 6, model_nv) = dJ_hd_r;
			kin_tasks_walk.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

			auto resLeg = pinKinDynIn.computeInK_Hand(hd_l_rot_des, hd_l_pos_L_des, hd_r_rot_des, hd_r_pos_L_des);

			id = kin_tasks_walk.getId("HandTrackJoints");
			kin_tasks_walk.taskLib[id].errX = Eigen::VectorXd::Zero(14);
			kin_tasks_walk.taskLib[id].errX = resLeg.jointPosRes.block<14, 1>(0, 0) - q.block<14, 1>(7, 0);
			kin_tasks_walk.taskLib[id].derrX = Eigen::VectorXd::Zero(14);
			kin_tasks_walk.taskLib[id].ddxDes = Eigen::VectorXd::Zero(14);
			kin_tasks_walk.taskLib[id].dxDes = Eigen::VectorXd::Zero(14);
			kin_tasks_walk.taskLib[id].kp = Eigen::MatrixXd::Identity(14, 14) * 2000; //100
			kin_tasks_walk.taskLib[id].kd = Eigen::MatrixXd::Identity(14, 14) * 100;
			kin_tasks_walk.taskLib[id].J = Eigen::MatrixXd::Zero(14, model_nv);
			kin_tasks_walk.taskLib[id].J.block(0, 6, 14, 14) = Eigen::MatrixXd::Identity(14, 14);
			kin_tasks_walk.taskLib[id].dJ = Eigen::MatrixXd::Zero(14, model_nv);
			kin_tasks_walk.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);
		}

		if (fixedarm)
		{
			//fixed arm
			id = kin_tasks_walk.getId("FixedArm");
			auto& task = kin_tasks_walk.taskLib[id];

			// 1. 设置期望关节角度 (q_des)
			// 我们可以使用机器人的初始站立姿态作为固定的目标姿态。
			// qIniDes 是在 main 函数中通过逆解站立姿态计算得到的。
			Eigen::VectorXd q_arm_des = qIniDes.block<14, 1>(7, 0);

			// 2. 计算误差 (期望角度 - 实际角度)
			task.errX = q_arm_des - q.block<14, 1>(7, 0);

			// 3. 计算误差导数 (期望速度为0 - 实际速度)
			task.derrX = -dq.block<14, 1>(6, 0);

			// 4. 设置PD控制器增益
			// 使用非常高的增益来实现“刚性”锁定效果
			task.kp = Eigen::MatrixXd::Identity(14, 14) * 5000;
			task.kd = Eigen::MatrixXd::Identity(14, 14) * 200;

			// 5. 定义雅可比矩阵 J
			// 这是一个纯关节空间任务，所以雅可比矩阵是一个选择矩阵
			task.J = Eigen::MatrixXd::Zero(14, model_nv);
			// q向量的前6维是浮动基座，所以关节从第7个元素开始 (索引为6)
			task.J.block<14, 14>(0, 6) = Eigen::MatrixXd::Identity(14, 14);

			// 6. 其他设置
			task.ddxDes = Eigen::VectorXd::Zero(14);
			task.dxDes = Eigen::VectorXd::Zero(14);
			task.dJ = Eigen::MatrixXd::Zero(14, model_nv);
			task.W.diagonal() = Eigen::VectorXd::Ones(model_nv);
		}


		if (coordinate && delayed == 10000)
		{
//			kin_tasks_walk.addTask("KneeThighCoordination");
			std::vector<std::string> taskOrder_walk;
			taskOrder_walk.emplace_back("RedundantJoints");
			taskOrder_walk.emplace_back("static_Contact");
			if (fixedarm) taskOrder_walk.emplace_back("FixedArm");
			if (!fixedarm) taskOrder_walk.emplace_back("HandTrackJoints");
			taskOrder_walk.emplace_back("PosRot");
			taskOrder_walk.emplace_back("KneeThighCoordination");
			taskOrder_walk.emplace_back("SwingLeg");
			kin_tasks_walk.buildPriority(taskOrder_walk);
		}
		// --- 开始定义新的时延约束任务 ---
		if (coordinate)
		{
			uint8_t state = 0;
			int id = kin_tasks_walk.getId("KneeThighCoordination");
			auto& task = kin_tasks_walk.taskLib[id];

			// 初始化任务为1维
			task.errX = Eigen::VectorXd::Zero(1);
			task.derrX = Eigen::VectorXd::Zero(1);
			task.J = Eigen::MatrixXd::Zero(1, model_nv);
			task.dJ = Eigen::MatrixXd::Zero(1, model_nv);

			// 从缓冲区尾部获取0.3秒前的历史数据
			double l_hip_q_delayed = l_hip_q_hist.back();
			double l_hip_dq_delayed = l_hip_dq_hist.back();
			double r_hip_q_delayed = r_hip_q_hist.back();
			double r_hip_dq_delayed = r_hip_dq_hist.back();
//			serial1.sendFormattedData("%f,%f,%f,%f\r\n",l_hip_q_delayed,q(L_HIP_IDX),r_hip_q_delayed,q(R_HIP_IDX));

			// 计算当前步速 v (这里我们假设 v 是期望速度在x-y平面的模长)
			double v = curV_W.head<2>().norm();
//			v = 1.0;
			// --- 为左右腿分别计算目标值和误差 ---
			double q_target_l, q_target_r, dq_target_l, dq_target_r;

			r_hip_q_delayed = 2 * ((r_hip_q_delayed) * 180.0 / 3.1415926 - (-20.0)) / 55.0 - 1;
			// 右腿目标位置 q_target_r
			q_target_r = -0.7104 + 0.0124 * v
				- 0.8533 * r_hip_q_delayed
				+ 0.0595 * v * r_hip_q_delayed
				+ 1.7419 * std::pow(r_hip_q_delayed, 2)
				- 0.0622 * v * std::pow(r_hip_q_delayed, 2);
			q_target_r = -((q_target_r + 1) * 70 / 2 + (-5)) * 3.1415926 / 180.0 - 0.24;
//			q_target_r = -((q_target_r + 1) * 70 / 2 + (-5)) * 3.1415926 / 180.0;

			l_hip_q_delayed = 2 * ((l_hip_q_delayed) * 180.0 / 3.1415926 - (-20.0)) / 55.0 - 1; //归一化
			// 左腿目标位置 q_target_l
			q_target_l = -0.7104 + 0.0124 * v
				- 0.8533 * l_hip_q_delayed
				+ 0.0595 * v * l_hip_q_delayed
				+ 1.7419 * std::pow(l_hip_q_delayed, 2)
				- 0.0622 * v * std::pow(l_hip_q_delayed, 2);
//			serial1.sendFormattedData("%f,%f,%f\r\n",l_hip_q_hist.back(),l_hip_q_delayed,q_target_l);
			q_target_l = -((q_target_l + 1) * 70 / 2 + (-5)) * 3.1415926 / 180.0 - 0.24; //反归一化
//			q_target_l = -((q_target_l + 1) * 70 / 2 + (-5)) * 3.1415926 / 180.0;

			static float coordination_task_weight;
			// 平滑增加权重，直到为1
			if (coordination_task_weight < 1.0) {
				coordination_task_weight += timeStep / 0.5; // 0.5秒内从0到1
			}

			if (legStateCur == DataBus::LSt) // 左腿支撑，右腿摆动
			{
				task.errX(0) = q_target_r - q(R_KNEE_IDX);
				task.J(0, L_KNEE_IDX) = 0.0;
				task.J(0, R_KNEE_IDX) = 1.0;
//				task.dJ(0, L_KNEE_IDX) = 0.0;
//				task.dJ(0, R_KNEE_IDX) = 0.0;
				printf("___________Left leg stance___________\r\n");
				state = 1;
			}
			else if (legStateCur == DataBus::RSt)
			{
				task.errX(0) = q_target_l - q(L_KNEE_IDX);
				task.J(0, L_KNEE_IDX) = 1.0;
				task.J(0, R_KNEE_IDX) = 0.0;
//				task.dJ(0, L_KNEE_IDX) = 0.0;
//				task.dJ(0, R_KNEE_IDX) = 0.0;
				printf("___________Right leg stance___________\r\n");
				state = 0;
			}
//			task.errX = task.errX * coordination_task_weight;
			// 设置控制器参数
			task.ddxDes = Eigen::VectorXd::Zero(1);
			task.dxDes = Eigen::VectorXd::Zero(1);
			task.kp = Eigen::MatrixXd::Identity(1, 1) * 3000; // 保持较高增益以强制执行
			task.kd = Eigen::MatrixXd::Identity(1, 1) * 0;
			task.W.diagonal() = Eigen::VectorXd::Ones(model_nv);
		}
	}

//	printcounter++;
//	if(printcounter == 5)
//	{
//		printf("%f,%f,%f,%f,%f\r\n",q_target_l,q(L_KNEE_IDX),q_target_r,q(R_KNEE_IDX),v);
//		printf("%d\r\n",model_nv);
//		serial1.sendFormattedData("%f,%f,%f,%f,%f,%f,%f,%d\r\n",q(L_HIP_IDX),q(R_HIP_IDX),-q_target_l,-q(L_KNEE_IDX),-q_target_r,-q(R_KNEE_IDX),v,state);
//		serial1.sendFormattedData("%f,%f,%f,%f\r\n",q(L_HIP_IDX),q(R_HIP_IDX),-q(L_KNEE_IDX),-q(R_KNEE_IDX));
//		printcounter = 0;
//	}

	/// -------- stand -------------
	{
//        int id = kin_tasks_stand.getId("static_Contact");
//        kin_tasks_stand.taskLib[id].errX = Eigen::VectorXd::Zero(12);
//        kin_tasks_stand.taskLib[id].derrX = Eigen::VectorXd::Zero(12);
//        kin_tasks_stand.taskLib[id].ddxDes = Eigen::VectorXd::Zero(12);
//        kin_tasks_stand.taskLib[id].dxDes = Eigen::VectorXd::Zero(12);
//        kin_tasks_stand.taskLib[id].kp = Eigen::MatrixXd::Identity(12, 12) * 0;
//        kin_tasks_stand.taskLib[id].kd = Eigen::MatrixXd::Identity(12, 12) * 0;
//        kin_tasks_stand.taskLib[id].J=Jfe;
//        kin_tasks_stand.taskLib[id].J.block(0,22,12,3).setZero(); // exculde waist joints
//        kin_tasks_stand.taskLib[id].dJ = dJfe;
//        kin_tasks_stand.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		int id = kin_tasks_stand.getId("static_Contact");
		kin_tasks_stand.taskLib[id].errX = Eigen::VectorXd::Zero(12);
		kin_tasks_stand.taskLib[id].derrX = Eigen::VectorXd::Zero(12);
		kin_tasks_stand.taskLib[id].ddxDes = Eigen::VectorXd::Zero(12);
		kin_tasks_stand.taskLib[id].dxDes = Eigen::VectorXd::Zero(12);
		kin_tasks_stand.taskLib[id].kp = Eigen::MatrixXd::Identity(12, 12) * 0;
		kin_tasks_stand.taskLib[id].kd = Eigen::MatrixXd::Identity(12, 12) * 0;
		kin_tasks_stand.taskLib[id].J = Eigen::MatrixXd::Zero(12, model_nv);
		Eigen::MatrixXd taskCtMap = Eigen::MatrixXd::Zero(3, 3);
		taskCtMap(0, 0) = 0;
		taskCtMap(1, 1) = 1;
		taskCtMap(2, 2) = 1;
		taskCtMap = fe_l_rot_cur_W * taskCtMap * fe_l_rot_cur_W.transpose(); // disable ankle roll joint
		kin_tasks_stand.taskLib[id].J = Jfe;
		kin_tasks_stand.taskLib[id].J.block(3, 0, 3, model_nv) =
			taskCtMap * kin_tasks_stand.taskLib[id].J.block(3, 0, 3, model_nv);
		kin_tasks_stand.taskLib[id].J.block(9, 0, 3, model_nv) =
			taskCtMap * kin_tasks_stand.taskLib[id].J.block(9, 0, 3, model_nv);
		kin_tasks_stand.taskLib[id].J.block(0, 22, 12, 3).setZero(); // exculde waist joints
		kin_tasks_stand.taskLib[id].dJ = Eigen::MatrixXd::Zero(12, model_nv);
		kin_tasks_stand.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		id = kin_tasks_stand.getId("HipRPY");
		kin_tasks_stand.taskLib[id].errX = Eigen::VectorXd::Zero(3);
		Eigen::Matrix3d desRot = eul2Rot(0, 0, 0);
		kin_tasks_stand.taskLib[id].errX.block<3, 1>(0, 0) = diffRot(hip_link_rot, desRot);
		kin_tasks_stand.taskLib[id].derrX = Eigen::VectorXd::Zero(3);
		kin_tasks_stand.taskLib[id].ddxDes = Eigen::VectorXd::Zero(3);
		kin_tasks_stand.taskLib[id].dxDes = Eigen::VectorXd::Zero(3);
		kin_tasks_stand.taskLib[id].kp = Eigen::MatrixXd::Identity(3, 3) * 1000;
		kin_tasks_stand.taskLib[id].kd = Eigen::MatrixXd::Identity(3, 3) * 50;
		Eigen::MatrixXd taskMapRPY = Eigen::MatrixXd::Zero(3, 6);
		taskMapRPY(0, 3) = 1;
		taskMapRPY(1, 4) = 1;
		taskMapRPY(2, 5) = 1;
		kin_tasks_stand.taskLib[id].J = taskMapRPY * J_hip_link;
		kin_tasks_stand.taskLib[id].J.block(0, 22, 3, 3).setZero();
		kin_tasks_stand.taskLib[id].J.block(0, 6, 3, 14).setZero();
		kin_tasks_stand.taskLib[id].dJ = Eigen::MatrixXd::Zero(3, model_nv);
		kin_tasks_stand.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		id = kin_tasks_stand.getId("Pz");
		kin_tasks_stand.taskLib[id].errX = Eigen::VectorXd::Zero(1);
		kin_tasks_stand.taskLib[id].errX(0) = base_pos_des(2) - q(2);
		kin_tasks_stand.taskLib[id].derrX = Eigen::VectorXd::Zero(1);
		kin_tasks_stand.taskLib[id].ddxDes = Eigen::VectorXd::Zero(1);
		kin_tasks_stand.taskLib[id].dxDes = Eigen::VectorXd::Zero(1);
		kin_tasks_stand.taskLib[id].kp = Eigen::MatrixXd::Identity(1, 1) * 2000; //100
		kin_tasks_stand.taskLib[id].kd = Eigen::MatrixXd::Identity(1, 1) * 10;
		Eigen::MatrixXd taskMap = Eigen::MatrixXd::Zero(1, 6);
		taskMap(0, 2) = 1;
		kin_tasks_stand.taskLib[id].J = taskMap * J_base;
		kin_tasks_stand.taskLib[id].J.block(0, 22, 1, 3).setZero();
		kin_tasks_stand.taskLib[id].dJ = taskMap * dJ_base;
		kin_tasks_stand.taskLib[id].dJ.block(0, 22, 1, 3).setZero();
		kin_tasks_stand.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		id = kin_tasks_stand.getId("CoMTrack");
		kin_tasks_stand.taskLib[id].errX = Eigen::VectorXd::Zero(2);
		kin_tasks_stand.taskLib[id].errX = pCoMDes.block(0, 0, 2, 1) - pCoMCur.block(0, 0, 2, 1);
		kin_tasks_stand.taskLib[id].derrX = Eigen::VectorXd::Zero(2);
		kin_tasks_stand.taskLib[id].ddxDes = Eigen::VectorXd::Zero(2);
		kin_tasks_stand.taskLib[id].dxDes = Eigen::VectorXd::Zero(2);
		kin_tasks_stand.taskLib[id].kp = Eigen::MatrixXd::Identity(2, 2) * 2000; //100
		kin_tasks_stand.taskLib[id].kd = Eigen::MatrixXd::Identity(2, 2) * 100;
		kin_tasks_stand.taskLib[id].J = Jcom.block(0, 0, 2, model_nv);
		kin_tasks_stand.taskLib[id].J.block(0, 6, 2, 14).setZero();
		kin_tasks_stand.taskLib[id].dJ = Eigen::MatrixXd::Zero(2, model_nv);
		kin_tasks_stand.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);
//        std::cout<<"pCoMCur"<<std::endl<<pCoMCur.transpose()<<std::endl;
//        std::cout<<"pCoMDes"<<std::endl<<pCoMDes.transpose()<<std::endl;

		id = kin_tasks_stand.getId("CoMXY_HipRPY");
		taskMapRPY = Eigen::MatrixXd::Zero(3, 6);
		taskMapRPY(0, 3) = 1;
		taskMapRPY(1, 4) = 1;
		taskMapRPY(2, 5) = 1;
		kin_tasks_stand.taskLib[id].errX = Eigen::VectorXd::Zero(5);
		kin_tasks_stand.taskLib[id].errX.block(0, 0, 2, 1) = pCoMDes.block(0, 0, 2, 1) - pCoMCur.block(0, 0, 2, 1);
//            kin_tasks_stand.taskLib[id].errX[0]+=0.01;
		desRot = eul2Rot(base_rpy_des(0), base_rpy_des(1), base_rpy_des(2));
		kin_tasks_stand.taskLib[id].errX.block<3, 1>(2, 0) = diffRot(hip_link_rot, desRot);
		kin_tasks_stand.taskLib[id].derrX = Eigen::VectorXd::Zero(5);
//            kin_tasks_stand.taskLib[id].derrX.block(0,0,2,1)=-(Jcom*dq).block(0,0,2,1);
//            kin_tasks_stand.taskLib[id].derrX.block(2,0,3,1)=-taskMapRPY*J_hip_link*dq;
		kin_tasks_stand.taskLib[id].ddxDes = Eigen::VectorXd::Zero(5);
		kin_tasks_stand.taskLib[id].dxDes = Eigen::VectorXd::Zero(5);
		kin_tasks_stand.taskLib[id].kp = Eigen::MatrixXd::Identity(5, 5) * 1000; //100
		kin_tasks_stand.taskLib[id].kd = Eigen::MatrixXd::Identity(5, 5) * 10;
		kin_tasks_stand.taskLib[id].kp.block(2, 2, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1000; //100 // for hip rpy
		kin_tasks_stand.taskLib[id].kd.block(2, 2, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 10; //100  // for hip rpy
		kin_tasks_stand.taskLib[id].J = Eigen::MatrixXd::Zero(5, model_nv);
		kin_tasks_stand.taskLib[id].J.block(0, 0, 2, model_nv) = Jcom.block(0, 0, 2, model_nv);
		kin_tasks_stand.taskLib[id].J.block(2, 0, 3, model_nv) = taskMapRPY * J_hip_link;
		kin_tasks_stand.taskLib[id].J.block(2, 22, 3, 3).setZero(); // exculde waist joints
		kin_tasks_stand.taskLib[id].J.block(2, 6, 3, 14).setZero(); // exculde arm joints
		//kin_tasks_stand.taskLib[id].J.block(0,6,2,14).setZero();
		kin_tasks_stand.taskLib[id].dJ = Eigen::MatrixXd::Zero(5, model_nv);
		kin_tasks_stand.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);
		kin_tasks_stand.taskLib[id].W.diagonal()(22) = 200;
		kin_tasks_stand.taskLib[id].W.diagonal()(23) = 200;

		// define swing arm motion
		Eigen::Vector3d hd_l_eul_L_des = { -1.253, 0.122, -1.732 };
		Eigen::Vector3d hd_r_eul_L_des = { 1.253, 0.122, 1.732 };
		Eigen::Matrix3d hd_l_rot_des = eul2Rot(hd_l_eul_L_des(0), hd_l_eul_L_des(1), hd_l_eul_L_des(2));
		Eigen::Matrix3d hd_r_rot_des = eul2Rot(hd_r_eul_L_des(0), hd_r_eul_L_des(1), hd_r_eul_L_des(2));

		Eigen::Vector3d base2shoulder_l_pos_L_des = { 0.0040, 0.1616, 0.3922 };
		Eigen::Vector3d shoulder2hand_l_pos_L_des = { -0.0240, 0.1584, -0.5512 };
		Eigen::Vector3d base2shoulder_r_pos_L_des = { 0.0040, -0.1616, 0.3922 };
		Eigen::Vector3d shoulder2hand_r_pos_L_des = { -0.0240, -0.1584, -0.5512 };
		double k = 1;
		hd_l_rot_des =
			eul2Rot(0, -k * r_shoulder_pitch, 0) * eul2Rot(hd_l_eul_L_des(0), hd_l_eul_L_des(1), hd_l_eul_L_des(2));
		hd_r_rot_des =
			eul2Rot(0, -k * l_shoulder_pitch, 0) * eul2Rot(hd_r_eul_L_des(0), hd_r_eul_L_des(1), hd_r_eul_L_des(2));
		Eigen::Vector3d hd_l_pos_L_des =
			eul2Rot(0, -k * r_shoulder_pitch, 0) * shoulder2hand_l_pos_L_des + base2shoulder_l_pos_L_des;//base_pos;
		Eigen::Vector3d hd_r_pos_L_des =
			eul2Rot(0, -k * l_shoulder_pitch, 0) * shoulder2hand_r_pos_L_des + base2shoulder_r_pos_L_des; //+ base_pos;

		auto resLeg = pinKinDynIn.computeInK_Hand(hd_l_rot_des, hd_l_pos_L_des, hd_r_rot_des, hd_r_pos_L_des);

		id = kin_tasks_stand.getId("HandTrackJoints");
		kin_tasks_stand.taskLib[id].errX = Eigen::VectorXd::Zero(14);
		kin_tasks_stand.taskLib[id].errX = resLeg.jointPosRes.block<14, 1>(0, 0) - q.block<14, 1>(7, 0);
		kin_tasks_stand.taskLib[id].derrX = Eigen::VectorXd::Zero(14);
		kin_tasks_stand.taskLib[id].ddxDes = Eigen::VectorXd::Zero(14);
		kin_tasks_stand.taskLib[id].dxDes = Eigen::VectorXd::Zero(14);
		kin_tasks_stand.taskLib[id].kp = Eigen::MatrixXd::Identity(14, 14) * 2000; //100
		kin_tasks_stand.taskLib[id].kd = Eigen::MatrixXd::Identity(14, 14) * 100;
		kin_tasks_stand.taskLib[id].J = Eigen::MatrixXd::Zero(14, model_nv);
		kin_tasks_stand.taskLib[id].J.block(0, 6, 14, 14) = Eigen::MatrixXd::Identity(14, 14);
		kin_tasks_stand.taskLib[id].dJ = Eigen::MatrixXd::Zero(14, model_nv);
		kin_tasks_stand.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		// Enter here functions to send actuator commands, like:
		// arm-l: 0-6, arm-r: 7-13, head: 14,15, waist: 16-18, leg-l: 19-24, leg-r: 25-30

		id = kin_tasks_stand.getId("HeadRP");
		kin_tasks_stand.taskLib[id].errX = Eigen::VectorXd::Zero(2);
		kin_tasks_stand.taskLib[id].errX(0) = 0 - q(21);
		kin_tasks_stand.taskLib[id].errX(1) = base_rpy_cur(1) - q(22);
		kin_tasks_stand.taskLib[id].derrX = Eigen::VectorXd::Zero(2);
		kin_tasks_stand.taskLib[id].ddxDes = Eigen::VectorXd::Zero(2);
		kin_tasks_stand.taskLib[id].dxDes = Eigen::VectorXd::Zero(2);
		kin_tasks_stand.taskLib[id].kp = Eigen::MatrixXd::Identity(2, 2) * 100; //100
		kin_tasks_stand.taskLib[id].kd = Eigen::MatrixXd::Identity(2, 2) * 10;
		kin_tasks_stand.taskLib[id].J = Eigen::MatrixXd::Zero(2, model_nv);
		kin_tasks_stand.taskLib[id].J(0, 20) = 1;
		kin_tasks_stand.taskLib[id].J(1, 21) = 1;
		kin_tasks_stand.taskLib[id].dJ = Eigen::MatrixXd::Zero(2, model_nv);
		kin_tasks_stand.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		id = kin_tasks_stand.getId("Roll_Pitch_Yaw");
		kin_tasks_stand.taskLib[id].errX = Eigen::VectorXd::Zero(3);
		desRot = eul2Rot(base_rpy_des(0), base_rpy_des(1), base_rpy_des(2));
		kin_tasks_stand.taskLib[id].errX = diffRot(base_rot, desRot);
		kin_tasks_stand.taskLib[id].derrX = Eigen::VectorXd::Zero(3);
		kin_tasks_stand.taskLib[id].derrX = -dq.block<3, 1>(3, 0);
		kin_tasks_stand.taskLib[id].ddxDes = Eigen::VectorXd::Zero(3);
		kin_tasks_stand.taskLib[id].dxDes = Eigen::VectorXd::Zero(3);
		kin_tasks_stand.taskLib[id].kp = Eigen::MatrixXd::Identity(3, 3) * 2000;
		kin_tasks_stand.taskLib[id].kd = Eigen::MatrixXd::Identity(3, 3) * 100;
		taskMap = Eigen::MatrixXd::Zero(3, 6);
		taskMap(0, 3) = 1;
		taskMap(1, 4) = 1;
		taskMap(2, 5) = 1;
		kin_tasks_stand.taskLib[id].J = taskMap * J_base;
		kin_tasks_stand.taskLib[id].dJ = taskMap * dJ_base;
		kin_tasks_stand.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);

		id = kin_tasks_stand.getId("fixedWaist");
		kin_tasks_stand.taskLib[id].errX = Eigen::VectorXd::Zero(3);
		kin_tasks_stand.taskLib[id].errX(0) = 0 - q(23);
		kin_tasks_stand.taskLib[id].errX(1) = 0 - q(24);
		kin_tasks_stand.taskLib[id].errX(2) = 0 - q(25);
		kin_tasks_stand.taskLib[id].derrX = Eigen::VectorXd::Zero(3);
		kin_tasks_stand.taskLib[id].ddxDes = Eigen::VectorXd::Zero(3);
		kin_tasks_stand.taskLib[id].dxDes = Eigen::VectorXd::Zero(3);
		kin_tasks_stand.taskLib[id].kp = Eigen::MatrixXd::Identity(3, 3) * 200;
		kin_tasks_stand.taskLib[id].kd = Eigen::MatrixXd::Identity(3, 3) * 20;
		kin_tasks_stand.taskLib[id].J = Eigen::MatrixXd::Zero(3, model_nv);
		kin_tasks_stand.taskLib[id].J(0, 22) = 1;
		kin_tasks_stand.taskLib[id].J(1, 23) = 1;
		kin_tasks_stand.taskLib[id].J(2, 24) = 1;
		kin_tasks_stand.taskLib[id].dJ = Eigen::MatrixXd::Zero(3, model_nv);
		kin_tasks_stand.taskLib[id].W.diagonal() = Eigen::VectorXd::Ones(model_nv);
	}

	if (motionStateCur == DataBus::Walk || motionStateCur == DataBus::Walk2Stand)
	{
		kin_tasks_walk.computeAll(des_delta_q, des_dq, des_ddq, dyn_M, dyn_M_inv, dq);
		delta_q_final_kin = kin_tasks_walk.out_delta_q;
		dq_final_kin = kin_tasks_walk.out_dq;
		ddq_final_kin = kin_tasks_walk.out_ddq;
	}
	else if (motionStateCur == DataBus::Stand)
	{
		kin_tasks_stand.computeAll(des_delta_q, des_dq, des_ddq, dyn_M, dyn_M_inv, dq);
		delta_q_final_kin = kin_tasks_stand.out_delta_q;
		dq_final_kin = kin_tasks_stand.out_dq;
		ddq_final_kin = kin_tasks_stand.out_ddq;
	}
	else
	{
		delta_q_final_kin = Eigen::VectorXd::Zero(model_nv);
		dq_final_kin = Eigen::VectorXd::Zero(model_nv);
		ddq_final_kin = Eigen::VectorXd::Zero(model_nv);
	}
	// final WBC output collection
}

void WBC_priority::copy_Eigen_to_real_t(qpOASES::real_t* target, const Eigen::MatrixXd& source, int nRows, int nCols)
{
	int count = 0;

	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			target[count++] = isinf(source(i, j)) ? qpOASES::INFTY : source(i, j);
		}
	}
}

void WBC_priority::setQini(const Eigen::VectorXd& qIniDesIn, const Eigen::VectorXd& qIniCurIn)
{
	qIniDes = qIniDesIn;
	qIniCur = qIniCurIn;
}