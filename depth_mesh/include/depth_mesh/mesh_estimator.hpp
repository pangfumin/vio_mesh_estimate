#ifndef _MESH_ESTIMATOR_H_
#define _MESH_ESTIMATOR_H_

#include "depth_mesh/flame.h"
#include <common/kinematics/Transformation.hpp>
namespace flame {

    class MeshEstimator {
    public:
        MeshEstimator(int width, int height,
                      const Matrix3f& K, const Matrix3f& Kinv, const Vector4f& distort,
                      const Params& parameters = Params());

        void processFrame(const double time,
                          const okvis::kinematics::Transformation& T_WC,
                          const cv::Mat& img_gray, bool isKeyframe);


        std::shared_ptr<flame::Flame> sensor_;
    private:

        static uint64_t img_id_;
        // Depth sensor.
        cv::Mat Kcv_, Dcv_;
        flame::Params params_;
        int poseframe_subsample_factor_;
    };
}
#endif