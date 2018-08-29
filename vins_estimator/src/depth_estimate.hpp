#ifndef  _DEPTH_ESTIMATE_H_
#define  _DEPTH_ESTIMATE_H_

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PolygonMesh.h>

#include <image_transport/image_transport.h>



#include <depth_mesh/flame.h>
#include <depth_mesh/utils/image_utils.h>
#include <depth_mesh/utils/stats_tracker.h>
#include <depth_mesh/utils/load_tracker.h>

class DepthEstimate  {
public:
    /**
     * @brief Constructor.
     */
    DepthEstimate();
    ~DepthEstimate() = default;

    DepthEstimate(const DepthEstimate& rhs) = delete;
    DepthEstimate& operator=(const DepthEstimate& rhs) = delete;

    DepthEstimate(DepthEstimate&& rhs) = delete;
    DepthEstimate& operator=(DepthEstimate&& rhs) = delete;


    /**
     * @brief Main processing loop.
     */
    void estimate();


private:
    // Keeps track of stats and load.
    fu::StatsTracker stats_;
    fu::LoadTracker load_;

    ros::NodeHandle nh_;
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
    std::shared_ptr<ros_sensor_streams::ASLRGBDOfflineStream> input_;
    Eigen::Matrix3f Kinv_;

    // Stuff for checking angular rates.
    float max_angular_rate_;
    double prev_time_;
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

#endif