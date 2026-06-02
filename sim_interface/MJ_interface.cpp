/*
This is part of OpenLoong Dynamics Control, an open project for the control of biped robot,
Copyright (C) 2024 Humanoid Robot (Shanghai) Co., Ltd, under Apache 2.0.
Feel free to use in any purpose, and cite OpenLoong-Dynamics-Control in any style, to contribute to the advancement of the community.
 <https://atomgit.com/openloong/openloong-dyn-control.git>
 <web@openloong.org.cn>
*/
#include "MJ_interface.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {
double getEnvDouble(const char *name, double default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }
    char *end = nullptr;
    const double parsed = std::strtod(value, &end);
    return end != value ? parsed : default_value;
}

double filteredContactForce(double raw_force, double previous_force,
                          double time_step, double filter_tc,
                          double clamp_force) {
    const double nonnegative = std::max(raw_force, 0.0);
    const double clamped =
        clamp_force > 0.0 ? std::min(nonnegative, clamp_force) : nonnegative;
    if (filter_tc <= 0.0) {
        return clamped;
    }
    const double alpha = time_step / (filter_tc + time_step);
    return previous_force + alpha * (clamped - previous_force);
}

double contactFrameForceWorldZ(const mjContact &contact,
                               const mjtNum contact_force[6]) {
    return contact.frame[2] * contact_force[0] +
           contact.frame[5] * contact_force[1] +
           contact.frame[8] * contact_force[2];
}

bool isFootBody(const mjModel *model, int geom, int foot_body_id_a,
                int foot_body_id_b) {
    const int body_id = model->geom_bodyid[geom];
    return body_id == foot_body_id_a || body_id == foot_body_id_b;
}

double footContactWorldFz(const mjModel *model, const mjData *data,
                          int foot_body_id_a, int foot_body_id_b) {
    double fz = 0.0;
    for (int contact_id = 0; contact_id < data->ncon; ++contact_id) {
        const mjContact &contact = data->contact[contact_id];
        if (contact.exclude || contact.efc_address < 0) {
            continue;
        }
        const int geom0 = contact.geom[0] >= 0 ? contact.geom[0] : contact.geom1;
        const int geom1 = contact.geom[1] >= 0 ? contact.geom[1] : contact.geom2;
        if (geom0 < 0 || geom1 < 0) {
            continue;
        }
        const bool foot_is_geom0 =
            isFootBody(model, geom0, foot_body_id_a, foot_body_id_b);
        const bool foot_is_geom1 =
            isFootBody(model, geom1, foot_body_id_a, foot_body_id_b);
        if (!foot_is_geom0 && !foot_is_geom1) {
            continue;
        }

        mjtNum contact_force[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        mj_contactForce(model, data, contact_id, contact_force);

        // MuJoCo's contact normal points from geom[0] to geom[1]. The contact
        // force returned here is expressed in that contact frame. We use the
        // world vertical support magnitude for the virtual foot force sensor.
        const double world_z_on_geom1 =
            contactFrameForceWorldZ(contact, contact_force);
        fz += std::abs(world_z_on_geom1);
    }
    return fz;
}

double sensorScalar(const mjModel *model, const mjData *data, int sensor_id) {
    if (sensor_id < 0) {
        return 0.0;
    }
    const int adr = model->sensor_adr[sensor_id];
    return adr >= 0 ? data->sensordata[adr] : 0.0;
}
} // namespace

MJ_Interface::MJ_Interface(mjModel *mj_modelIn, mjData *mj_dataIn) {
    mj_model=mj_modelIn;
    mj_data=mj_dataIn;
    timeStep=mj_model->opt.timestep;
    touchForceFilterTc = std::max(0.0, getEnvDouble("ODC_TOUCH_FORCE_FILTER_TC", touchForceFilterTc));
    touchForceClamp = std::max(0.0, getEnvDouble("ODC_TOUCH_FORCE_CLAMP", touchForceClamp));
    touchForceFilterTc = std::max(0.0, getEnvDouble("ODC_FOOT_FORCE_FILTER_TC", touchForceFilterTc));
    touchForceClamp = std::max(0.0, getEnvDouble("ODC_FOOT_FORCE_CLAMP", touchForceClamp));
    jointNum=JointName.size();
    jntId_qpos.assign(jointNum,0);
    jntId_qvel.assign(jointNum,0);
    jntId_dctl.assign(jointNum,0);
    motor_pos.assign(jointNum,0);
    motor_vel.assign(jointNum,0);
    motor_pos_Old.assign(jointNum,0);
    for (int i=0;i<jointNum;i++)
    {
        int tmpId= mj_name2id(mj_model,mjOBJ_JOINT,JointName[i].c_str());
        if (tmpId==-1)
        {
            std::cerr <<JointName[i]<< " not found in the XML file!" << std::endl;
            std::terminate();
        }
        jntId_qpos[i]=mj_model->jnt_qposadr[tmpId];
        jntId_qvel[i]=mj_model->jnt_dofadr[tmpId];
        std::string motorName=JointName[i];
        motorName="M"+motorName.substr(1);
        tmpId= mj_name2id(mj_model,mjOBJ_ACTUATOR,motorName.c_str());
        if (tmpId==-1)
        {
            std::cerr <<motorName<< " not found in the XML file!" << std::endl;
            std::terminate();
        }
        jntId_dctl[i]=tmpId;
    }
//    int adr = m->sensor_adr[sensorId];
//    int dim = m->sensor_dim[sensorId];
//    mjtNum sensor_data[dim];
//    mju_copy(sensor_data, &d->sensordata[adr], dim);
    baseBodyId= mj_name2id(mj_model,mjOBJ_BODY, baseName.c_str());
    orientataionSensorId= mj_name2id(mj_model, mjOBJ_SENSOR, orientationSensorName.c_str());
    velSensorId= mj_name2id(mj_model,mjOBJ_SENSOR,velSensorName.c_str());
    gyroSensorId= mj_name2id(mj_model,mjOBJ_SENSOR,gyroSensorName.c_str());
    accSensorId= mj_name2id(mj_model,mjOBJ_SENSOR,accSensorName.c_str());
    leftTouchSensorId= mj_name2id(mj_model,mjOBJ_SENSOR,leftTouchSensorName.c_str());
    rightTouchSensorId= mj_name2id(mj_model,mjOBJ_SENSOR,rightTouchSensorName.c_str());
    leftFootPitchBodyId= mj_name2id(mj_model,mjOBJ_BODY,leftFootPitchBodyName.c_str());
    leftFootRollBodyId= mj_name2id(mj_model,mjOBJ_BODY,leftFootRollBodyName.c_str());
    rightFootPitchBodyId= mj_name2id(mj_model,mjOBJ_BODY,rightFootPitchBodyName.c_str());
    rightFootRollBodyId= mj_name2id(mj_model,mjOBJ_BODY,rightFootRollBodyName.c_str());

}

void MJ_Interface::updateSensorValues() {
    for (int i=0;i<jointNum;i++){
        motor_pos_Old[i]=motor_pos[i];
        motor_pos[i]=mj_data->qpos[jntId_qpos[i]];
        motor_vel[i]=mj_data->qvel[jntId_qvel[i]];
    }
    for (int i=0;i<4;i++)
        baseQuat[i]=mj_data->sensordata[mj_model->sensor_adr[orientataionSensorId]+i];
    double tmp=baseQuat[0];
    baseQuat[0]=baseQuat[1];
    baseQuat[1]=baseQuat[2];
    baseQuat[2]=baseQuat[3];
    baseQuat[3]=tmp;

    rpy[0]= atan2(2*(baseQuat[3]*baseQuat[0]+baseQuat[1]*baseQuat[2]),1-2*(baseQuat[0]*baseQuat[0]+baseQuat[1]*baseQuat[1]));
    rpy[1]= asin(2*(baseQuat[3]*baseQuat[1]-baseQuat[0]*baseQuat[2]));
    rpy[2]= atan2(2*(baseQuat[3]*baseQuat[2]+baseQuat[0]*baseQuat[1]),1-2*(baseQuat[1]*baseQuat[1]+baseQuat[2]*baseQuat[2]));

    for (int i=0;i<3;i++)
    {
        double posOld=basePos[i];
        basePos[i]=mj_data->xpos[3*baseBodyId+i];
        baseAcc[i]=mj_data->sensordata[mj_model->sensor_adr[accSensorId]+i];
        baseAngVel[i]=mj_data->sensordata[mj_model->sensor_adr[gyroSensorId]+i];
        baseLinVel[i]=(basePos[i]-posOld)/(mj_model->opt.timestep);
    }
    const double rawLeftContact =
        leftFootPitchBodyId >= 0 && leftFootRollBodyId >= 0
            ? footContactWorldFz(mj_model, mj_data, leftFootPitchBodyId,
                                 leftFootRollBodyId)
            : 0.0;
    const double rawRightContact =
        rightFootPitchBodyId >= 0 && rightFootRollBodyId >= 0
            ? footContactWorldFz(mj_model, mj_data, rightFootPitchBodyId,
                                 rightFootRollBodyId)
            : 0.0;
    rawContactFz[0] = rawLeftContact;
    rawContactFz[1] = rawRightContact;
    rawTouchForce[0] = sensorScalar(mj_model, mj_data, leftTouchSensorId);
    rawTouchForce[1] = sensorScalar(mj_model, mj_data, rightTouchSensorId);
    if (!touchForceFilterInitialized) {
        touchForceFilt[0] =
            touchForceClamp > 0.0
                ? std::min(std::max(rawLeftContact, 0.0), touchForceClamp)
                : std::max(rawLeftContact, 0.0);
        touchForceFilt[1] =
            touchForceClamp > 0.0
                ? std::min(std::max(rawRightContact, 0.0), touchForceClamp)
                : std::max(rawRightContact, 0.0);
        touchForceFilterInitialized = true;
    } else {
        touchForceFilt[0] =
            filteredContactForce(rawLeftContact, touchForceFilt[0], timeStep,
                               touchForceFilterTc, touchForceClamp);
        touchForceFilt[1] =
            filteredContactForce(rawRightContact, touchForceFilt[1], timeStep,
                               touchForceFilterTc, touchForceClamp);
    }
    f3d[0][0]=0.0;
    f3d[1][0]=0.0;
    f3d[2][0]= touchForceFilt[0];
    f3d[0][1]=0.0;
    f3d[1][1]=0.0;
    f3d[2][1]= touchForceFilt[1];

}

void MJ_Interface::setMotorsTorque(std::vector<double> &tauIn) {
    for (int i=0;i<jointNum;i++)
        mj_data->ctrl[i]=tauIn.at(i);
}

void MJ_Interface::dataBusWrite(DataBus &busIn) {
    busIn.motors_pos_cur=motor_pos;
    busIn.motors_vel_cur=motor_vel;
    busIn.rpy[0]=rpy[0];
    busIn.rpy[1]=rpy[1];
    busIn.rpy[2]=rpy[2];
    busIn.fL[0]=f3d[0][0];
    busIn.fL[1]=f3d[1][0];
    busIn.fL[2]=f3d[2][0];
    busIn.fR[0]=f3d[0][1];
    busIn.fR[1]=f3d[1][1];
    busIn.fR[2]=f3d[2][1];
    busIn.foot_contact_fz_raw_l = rawContactFz[0];
    busIn.foot_contact_fz_raw_r = rawContactFz[1];
    busIn.foot_contact_fz_l = f3d[2][0];
    busIn.foot_contact_fz_r = f3d[2][1];
    busIn.foot_touch_raw_l = rawTouchForce[0];
    busIn.foot_touch_raw_r = rawTouchForce[1];
    busIn.basePos[0]=basePos[0];
    busIn.basePos[1]=basePos[1];
    busIn.basePos[2]=basePos[2];
    busIn.baseLinVel[0]=baseLinVel[0];
    busIn.baseLinVel[1]=baseLinVel[1];
    busIn.baseLinVel[2]=baseLinVel[2];
    busIn.baseAcc[0]=baseAcc[0];
    busIn.baseAcc[1]=baseAcc[1];
    busIn.baseAcc[2]=baseAcc[2];
    busIn.baseAngVel[0]=baseAngVel[0];
    busIn.baseAngVel[1]=baseAngVel[1];
    busIn.baseAngVel[2]=baseAngVel[2];
    busIn.updateQ();
}

