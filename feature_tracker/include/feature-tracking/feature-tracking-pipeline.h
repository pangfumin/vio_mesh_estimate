#ifndef FEATURE_TRACKING_FEATURE_TRACKING_PIPELINE_H_
#define FEATURE_TRACKING_FEATURE_TRACKING_PIPELINE_H_

#include <string>
#include <vector>

#include <aslam/common/memory.h>
#include <aslam/common/statistics/accumulator.h>
#include <aslam/frames/feature-track.h>
#include <aslam/visualization/feature-track-visualizer.h>
#include <common/macros.h>

#include "feature-tracking/feature-detection-extraction.h"
#include "feature-tracking/feature-track-extractor.h"

namespace aslam {
class FeatureTrackerLk;
class VisualNFrame;
}



namespace aslam_cv_visualization {
class VisualNFrameFeatureTrackVisualizer;
}

namespace feature_tracking {

/// Pipeline to rerun tracking of features and triangulation of landmarks on a
/// already existing VIMap (i.e. with pose-graph, etc.).
/// Visualization of keypoints, keypoint matches and feature tracks is
/// available. See the flags at the top of the the source file.
class FeatureTrackingPipeline {
 public:
  MAPLAB_POINTER_TYPEDEFS(FeatureTrackingPipeline);
  MAPLAB_DISALLOW_EVIL_CONSTRUCTORS(FeatureTrackingPipeline);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FeatureTrackingPipeline();
  virtual ~FeatureTrackingPipeline() = default;

 protected:
  const std::string feature_tracking_ros_base_topic_;
  const bool visualize_keypoint_matches_;

 private:
  aslam_cv_visualization::VisualNFrameFeatureTrackVisualizer
      feature_track_visualizer_;

  vio_common::FeatureTrackExtractor::UniquePtr track_extractor_;

  bool processed_first_nframe_;

  statistics::Accumulator<size_t, size_t, statistics::kInfiniteWindowSize>
      successfully_triangulated_landmarks_accumulator_;
};
}  // namespace feature_tracking

#endif  // FEATURE_TRACKING_FEATURE_TRACKING_PIPELINE_H_
