#ifndef _MESH_ESTIMATOR_H_
#define _MESH_ESTIMATOR_H_

#include "depth_mesh/flame.h"
#include <common/kinematics/Transformation.hpp>


namespace flame {

    class MeshEstimator {
    public:
        MeshEstimator( int width, int height,
                      const Matrix3f& K0, const Matrix3f& K0inv,
                      const Vector4f& distort0,
                      const Matrix3f& K1, const Matrix3f& K1inv,
                      const Vector4f& distort1,
                      const Params& parameters = Params());

        void processFrame(const okvis::Time time, int64_t img_id,
                          const okvis::kinematics::Transformation& T_WC0,
                          const cv::Mat& img_gray0,
                          const okvis::kinematics::Transformation& T_WC1,
                          const cv::Mat& img_gray1, bool isKeyframe);


        std::shared_ptr<flame::Flame> sensor_;
    private:

        // Depth sensor.
        cv::Mat K0cv_, D0cv_;
        cv::Mat K1cv_, D1cv_;
        flame::Params params_;
        int poseframe_subsample_factor_;
    };
}
#endif