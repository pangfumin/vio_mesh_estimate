
#ifndef _TYPES_H_
#define _TYPES_H_
#include <common/Time/Time.hpp>
#include <Eigen/Core>

namespace common {
    struct State {
        Eigen::Vector3d P;
        Eigen::Vector3d V;
        Eigen::Matrix3d R;
        Eigen::Vector3d Ba;
        Eigen::Vector3d Bg;
        okvis::Time time;
    };
}


#endif