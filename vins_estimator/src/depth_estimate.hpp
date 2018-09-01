
#include <stdio.h>
#include <sys/stat.h>

#include <cstdio>
#include <ctime>
#include <csignal>

#include <memory>
#include <limits>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Core>

#include <common/kinematics/Transformation.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PolygonMesh.h>

#include <image_transport/image_transport.h>

#include <cv_bridge/cv_bridge.h>
#include <common/Time/Time.hpp>

#include <depth_mesh/flame.h>
#include <depth_mesh/utils/image_utils.h>
#include <depth_mesh/utils/stats_tracker.h>
#include <depth_mesh/utils/load_tracker.h>

#include "ros_utils.h"

namespace bfs = boost::filesystem;
namespace fu = flame::utils;

namespace flame {
/**
 * @brief Runs FLAME on a dataset of images.

 * Input is taken from data in ASL format:
 * http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
 *
 * Three ASL folders are needed:
 *   1. Pose data relative to a given world frame.
 *   2. RGB data
 *   3. Depthmap data (for comparison)
 */
    class DepthEstimate {
    public:
        /**
         * @brief Constructor.
         */
        DepthEstimate(ros::NodeHandle &pnh,
                Eigen::Matrix3f K0, Eigen::Matrix3f K1,
                int width, int height);

        ~DepthEstimate() = default;

        DepthEstimate(const DepthEstimate &rhs) = delete;

        DepthEstimate &operator=(const DepthEstimate &rhs) = delete;

        DepthEstimate(DepthEstimate &&rhs) = delete;

        DepthEstimate &operator=(DepthEstimate &&rhs) = delete;

        int updateFramePoses(const std::vector<common::State> vio_states,
                             const okvis::kinematics::Transformation T_SC0);
        void processFrame(const uint32_t img_id, const okvis::Time time,
                          const okvis::kinematics::Transformation &pose0,
                          const cv::Mat1b &img_gray0,
                          const okvis::kinematics::Transformation &pose1,
                          const cv::Mat1b &img_gray1,
                          bool asKeyframe);

    private:
        // Keeps track of stats and load.
        fu::StatsTracker stats_;
        fu::LoadTracker load_;

        ros::NodeHandle pnh_;

        // Number of images processed.
        int num_imgs_;

        // Target processing rate. Can artificially slow down processing so that ROS can
        // keep up.
        float rate_;

        // Frame ID of the camera in frame camera_world_frame_id.
        std::string camera_frame_id_;

        // Frame id of the world in camera (Right-Down-Forward) coordinates.
        std::string camera_world_frame_id_;

        int subsample_factor_; // Process one out of this many images.
        int poseframe_subsample_factor_; // Create a poseframe every this number of images.
        int resize_factor_;

        // Save truth stats.
        std::string output_dir_;
        bool pass_in_truth_; // Pass truth into processing.

        // Input stream object.

        int width_, height_;
        Eigen::Matrix3f K0_;
        Eigen::Matrix3f K0inv_;
        Eigen::Matrix3f K1_;
        Eigen::Matrix3f K1inv_;

        // Stuff for checking angular rates.
        float max_angular_rate_;
        okvis::Time prev_time_;
        okvis::kinematics::Transformation prev_pose_;

        // Depth sensor.
        flame::Params params_;
        std::shared_ptr<flame::Flame> sensor_;

        // Publishes mesh.
        bool publish_mesh_;
        ros::Publisher mesh_pub_;

        // Publishes depthmap and stuff.
        image_transport::ImageTransport it_;
        bool publish_idepthmap_;
        bool publish_depthmap_;
        bool publish_features_;
        image_transport::CameraPublisher idepth_pub_;
        image_transport::CameraPublisher depth_pub_;
        image_transport::CameraPublisher features_pub_;

        // Publish pointcloud.
        bool publish_cloud_;
        ros::Publisher cloud_pub_;

        // Publishes statistics.
        bool publish_stats_;
        ros::Publisher stats_pub_;
        ros::Publisher nodelet_stats_pub_;
        int load_integration_factor_;

        // Publishes debug images.
        image_transport::Publisher debug_wireframe_pub_;
        image_transport::Publisher debug_features_pub_;
        image_transport::Publisher debug_detections_pub_;
        image_transport::Publisher debug_matches_pub_;
        image_transport::Publisher debug_normals_pub_;
        image_transport::Publisher debug_idepthmap_pub_;
    };


}