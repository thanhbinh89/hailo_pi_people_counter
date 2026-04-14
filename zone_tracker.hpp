#pragma once

#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

// ─────────────────────────────────────────────────────────────────────────────
// Configuration — loaded from config.yaml
// ─────────────────────────────────────────────────────────────────────────────

struct AppConfig {
    // Zone: normalized coordinates (0.0 – 1.0)
    float zone_x1 = 0.2f;
    float zone_y1 = 0.2f;
    float zone_x2 = 0.8f;
    float zone_y2 = 0.8f;

    // Detection
    float confidence_threshold = 0.5f;

    // Centroid tracker
    int   max_disappeared = 30;
    float max_distance    = 100.0f;

    // Display
    bool show_zone         = true;
    bool show_tracking_ids = true;
    bool show_stats        = true;
};

AppConfig load_config(const std::string& path);

// ─────────────────────────────────────────────────────────────────────────────
// PersonRecord — state for one tracked individual
// ─────────────────────────────────────────────────────────────────────────────

struct PersonRecord {
    int           id;
    cv::Point2f   centroid;
    bool          in_zone         = false;
    double        total_time_in_zone = 0.0;   // seconds accumulated across zone visits
    std::chrono::steady_clock::time_point entry_time;  // when current zone visit started
    int           disappeared     = 0;        // consecutive frames with no matching detection
};

// ─────────────────────────────────────────────────────────────────────────────
// ZoneTracker — centroid-based multi-person tracker with zone analytics
//
// Thread safety: all methods must be called from the same thread
//                (the postprocess thread in the 3-thread pipeline).
// ─────────────────────────────────────────────────────────────────────────────

class ZoneTracker {
public:
    explicit ZoneTracker(const AppConfig& cfg);

    // Feed new person bounding boxes (pixel coords in original frame).
    // Call once per frame with all current detections.
    void update(const std::vector<cv::Rect>& detections, int frame_w, int frame_h);

    // Draw zone rectangle, person markers, and stats panel onto frame (BGR).
    void draw(cv::Mat& frame) const;

    // Counters
    int total_entries()   const { return m_total_entries;   }
    int total_exits()     const { return m_total_exits;     }
    int current_in_zone() const { return m_current_in_zone; }

private:
    AppConfig                m_cfg;
    std::map<int, PersonRecord> m_persons;
    int                      m_next_id       = 0;
    int                      m_total_entries = 0;
    int                      m_total_exits   = 0;
    int                      m_current_in_zone = 0;

    // Helpers
    cv::Rect get_zone_rect(int w, int h) const;
    bool     is_in_zone(const cv::Point2f& pt, int w, int h) const;
    double   current_dwell(const PersonRecord& pr) const;   // total + ongoing seconds
};
