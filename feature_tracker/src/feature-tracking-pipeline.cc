#include "feature-tracking/feature-tracking-pipeline.h"

#include <aslam/cameras/camera.h>
#include <aslam/common/pose-types.h>
#include <aslam/common/statistics/statistics.h>
#include <aslam/common/timer.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <aslam/geometric-vision/match-outlier-rejection-twopt.h>
#include <aslam/matcher/match.h>
#include <aslam/triangulation/triangulation.h>
#include <aslam/visualization/basic-visualization.h>
#include <aslam/visualization/feature-track-visualizer.h>
#include <glog/logging.h>
#include <common/progress-bar.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


DEFINE_bool(
    feature_tracker_visualize_feature_tracks, false,
    "Flag indicating whether the full feature tracks are visualized.");

DEFINE_bool(
    feature_tracker_visualize_keypoints, false,
    "Flag indicating whether keypoints are visualized.");

DEFINE_bool(
    feature_tracker_visualize_keypoints_individual_frames, false,
    "Flag indicating whether keypoints are visualized with each frame "
    "on a separate ROS topic.");

DEFINE_bool(
    feature_tracker_visualize_keypoint_matches, true,
    "Flag indicating whether keypoint matches are visualized.");

DEFINE_bool(
    feature_tracker_publish_raw_images, false,
    "Flag indicating whether the raw images are published over ROS.");

DEFINE_bool(
    feature_tracker_check_map_for_consistency, false,
    "Flag indicating whether the map is checked for consistency after "
    "rerunning the feature tracking.");

namespace feature_tracking {

FeatureTrackingPipeline::FeatureTrackingPipeline()
    : feature_tracking_ros_base_topic_("tracking/"),
      visualize_keypoint_matches_(
          FLAGS_feature_tracker_visualize_keypoint_matches),
      processed_first_nframe_(false) {}

}  // namespace feature_tracking
