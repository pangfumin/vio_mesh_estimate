#include "depth_mesh/mesh_estimator.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

namespace flame {

    uint64_t MeshEstimator::img_id_ = 0;
    MeshEstimator::MeshEstimator(int width, int height,
                                 const Matrix3f& K, const Matrix3f& Kinv,
                                 const Vector4f& distort,
                                 const Params& parameters):
            params_(parameters),
            poseframe_subsample_factor_(6) {

        cv::eigen2cv(K, Kcv_);
        cv::eigen2cv(distort, Dcv_);

        sensor_ = std::make_shared<flame::Flame>(width,
                                                 height,
                                                 K,
                                                 Kinv,
                                                 params_);

    }

    void MeshEstimator::processFrame( const double time,
                      const okvis::kinematics::Transformation& T_WC,
                                      const cv::Mat& img_gray, bool isKeyframe) {
//
        /*==================== Process image ====================*/
        cv::Mat img_gray_undist;
        cv::undistort(img_gray, img_gray_undist, Kcv_, Dcv_);


//        std::cout<< T_WC.T() << std::endl;
//        std::cout<< pose.unit_quaternion().toRotationMatrix() << std::endl;


        bool is_poseframe = isKeyframe;

        bool update_success = false;

        update_success = sensor_->update(time, img_id_, T_WC, img_gray_undist,
                                             is_poseframe);
        img_id_ ++;
//        if (!update_success) {
//            //ROS_WARN("FlameOffline: Unsuccessful update.\n");
//            return;
//        }
        Image3b wireImage = sensor_->getDebugImageWireframe();
//        Image3b wireImage = sensor_->getDebugImageFeatures();
        cv::imshow("wireImage", wireImage);

        Image3b depthImage = sensor_->getDebugImageInverseDepthMap();
        cv::imshow("depthImage", depthImage);
        cv::waitKey(2);
//
//        // todo : add publish and result into buffer

    }

}