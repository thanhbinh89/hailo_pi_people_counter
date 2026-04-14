#include "zone_tracker.hpp"

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Config loader
// ─────────────────────────────────────────────────────────────────────────────

AppConfig load_config(const std::string& path)
{
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load config: " + path + " | " + e.what());
    }

    AppConfig cfg;

    if (root["zone"]) {
        const auto& z = root["zone"];
        cfg.zone_x1 = z["x1"].as<float>(0.2f);
        cfg.zone_y1 = z["y1"].as<float>(0.2f);
        cfg.zone_x2 = z["x2"].as<float>(0.8f);
        cfg.zone_y2 = z["y2"].as<float>(0.8f);
    }
    if (root["detection"]) {
        cfg.confidence_threshold = root["detection"]["confidence_threshold"].as<float>(0.5f);
    }
    if (root["tracker"]) {
        cfg.max_disappeared = root["tracker"]["max_disappeared"].as<int>(30);
        cfg.max_distance    = root["tracker"]["max_distance"].as<float>(100.0f);
    }
    if (root["display"]) {
        cfg.show_zone         = root["display"]["show_zone"].as<bool>(true);
        cfg.show_tracking_ids = root["display"]["show_tracking_ids"].as<bool>(true);
        cfg.show_stats        = root["display"]["show_stats"].as<bool>(true);
    }

    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// ZoneTracker implementation
// ─────────────────────────────────────────────────────────────────────────────

ZoneTracker::ZoneTracker(const AppConfig& cfg) : m_cfg(cfg) {}

cv::Rect ZoneTracker::get_zone_rect(int w, int h) const
{
    int x1 = static_cast<int>(m_cfg.zone_x1 * w);
    int y1 = static_cast<int>(m_cfg.zone_y1 * h);
    int x2 = static_cast<int>(m_cfg.zone_x2 * w);
    int y2 = static_cast<int>(m_cfg.zone_y2 * h);
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

bool ZoneTracker::is_in_zone(const cv::Point2f& pt, int w, int h) const
{
    return get_zone_rect(w, h).contains(
        cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
}

double ZoneTracker::current_dwell(const PersonRecord& pr) const
{
    double t = pr.total_time_in_zone;
    if (pr.in_zone) {
        auto now = std::chrono::steady_clock::now();
        t += std::chrono::duration<double>(now - pr.entry_time).count();
    }
    return t;
}

// ─── update ──────────────────────────────────────────────────────────────────

void ZoneTracker::update(const std::vector<cv::Rect>& detections, int frame_w, int frame_h)
{
    auto now = std::chrono::steady_clock::now();

    // Compute centroids of incoming detections
    std::vector<cv::Point2f> new_centroids;
    new_centroids.reserve(detections.size());
    for (const auto& r : detections) {
        new_centroids.emplace_back(r.x + r.width  * 0.5f,
                                   r.y + r.height * 0.5f);
    }

    // ── No existing tracks: register everything ──────────────────────────────
    if (m_persons.empty()) {
        for (const auto& c : new_centroids) {
            PersonRecord pr;
            pr.id        = m_next_id++;
            pr.centroid  = c;
            pr.in_zone   = is_in_zone(c, frame_w, frame_h);
            if (pr.in_zone) {
                pr.entry_time = now;
                ++m_total_entries;
                ++m_current_in_zone;
            }
            m_persons[pr.id] = pr;
        }
        return;
    }

    // ── Build index arrays for existing tracks ────────────────────────────────
    std::vector<int>         person_ids;
    std::vector<cv::Point2f> person_centroids;
    person_ids.reserve(m_persons.size());
    person_centroids.reserve(m_persons.size());
    for (auto& [id, pr] : m_persons) {
        person_ids.push_back(id);
        person_centroids.push_back(pr.centroid);
    }

    std::vector<bool> det_used(new_centroids.size(), false);
    std::vector<bool> person_matched(person_ids.size(), false);

    // ── Greedy nearest-neighbour matching ─────────────────────────────────────
    // For each existing track, find closest unmatched detection
    for (size_t pi = 0; pi < person_ids.size(); ++pi) {
        float min_dist = std::numeric_limits<float>::max();
        int   best_di  = -1;

        for (size_t di = 0; di < new_centroids.size(); ++di) {
            if (det_used[di]) continue;
            float dx = person_centroids[pi].x - new_centroids[di].x;
            float dy = person_centroids[pi].y - new_centroids[di].y;
            float d  = std::sqrt(dx * dx + dy * dy);
            if (d < min_dist) { min_dist = d; best_di = static_cast<int>(di); }
        }

        int id  = person_ids[pi];
        auto& pr = m_persons[id];

        if (best_di >= 0 && min_dist <= m_cfg.max_distance) {
            // ── Matched ───────────────────────────────────────────────────────
            det_used[best_di]  = true;
            person_matched[pi] = true;
            pr.centroid    = new_centroids[best_di];
            pr.disappeared = 0;

            bool now_in = is_in_zone(pr.centroid, frame_w, frame_h);

            if (!pr.in_zone && now_in) {
                // Entered the zone
                pr.in_zone    = true;
                pr.entry_time = now;
                ++m_total_entries;
                ++m_current_in_zone;
            } else if (pr.in_zone && !now_in) {
                // Left the zone
                pr.total_time_in_zone +=
                    std::chrono::duration<double>(now - pr.entry_time).count();
                pr.in_zone = false;
                ++m_total_exits;
                --m_current_in_zone;
            }
        } else {
            // ── No match — increment disappearance counter ────────────────────
            ++pr.disappeared;
        }
    }

    // ── Retire tracks that have been missing too long ─────────────────────────
    std::vector<int> to_remove;
    for (auto& [id, pr] : m_persons) {
        if (pr.disappeared > m_cfg.max_disappeared) {
            if (pr.in_zone) {
                pr.total_time_in_zone +=
                    std::chrono::duration<double>(now - pr.entry_time).count();
                --m_current_in_zone;
                ++m_total_exits;
            }
            to_remove.push_back(id);
        }
    }
    for (int id : to_remove) { m_persons.erase(id); }

    // ── Register unmatched detections as new tracks ───────────────────────────
    for (size_t di = 0; di < new_centroids.size(); ++di) {
        if (det_used[di]) continue;
        PersonRecord pr;
        pr.id        = m_next_id++;
        pr.centroid  = new_centroids[di];
        pr.in_zone   = is_in_zone(pr.centroid, frame_w, frame_h);
        if (pr.in_zone) {
            pr.entry_time = now;
            ++m_total_entries;
            ++m_current_in_zone;
        }
        m_persons[pr.id] = pr;
    }
}

// ─── draw ─────────────────────────────────────────────────────────────────────

void ZoneTracker::draw(cv::Mat& frame) const
{
    const int w = frame.cols;
    const int h = frame.rows;

    // ── Zone rectangle ────────────────────────────────────────────────────────
    if (m_cfg.show_zone) {
        auto zone = get_zone_rect(w, h);
        cv::rectangle(frame, zone, cv::Scalar(0, 220, 255), 2);
        cv::putText(frame, "COUNT ZONE",
                    zone.tl() + cv::Point(5, 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(0, 220, 255), 2);
    }

    // ── Per-person markers ────────────────────────────────────────────────────
    if (m_cfg.show_tracking_ids) {
        for (const auto& [id, pr] : m_persons) {
            if (pr.disappeared > 0) continue;   // only draw currently visible persons

            // Green dot = inside zone, grey dot = outside
            cv::Scalar color = pr.in_zone ? cv::Scalar(0, 230, 0)
                                          : cv::Scalar(180, 180, 180);
            cv::Point  ctr(static_cast<int>(pr.centroid.x),
                           static_cast<int>(pr.centroid.y));

            cv::circle(frame, ctr, 6, color, -1);

            // Label: "ID:N  X.Xs" (dwell shown only while in zone)
            std::ostringstream label;
            label << "ID:" << id;
            if (pr.in_zone) {
                label << "  " << std::fixed << std::setprecision(1)
                      << current_dwell(pr) << "s";
            }
            cv::putText(frame, label.str(),
                        ctr + cv::Point(9, 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.48, color, 1);
        }
    }

    // ── Stats panel (top-left) ────────────────────────────────────────────────
    if (m_cfg.show_stats) {
        const int panel_w = 280;
        const int panel_h = 95;
        cv::rectangle(frame, cv::Rect(0, 0, panel_w, panel_h),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::rectangle(frame, cv::Rect(0, 0, panel_w, panel_h),
                      cv::Scalar(60, 60, 60), 1);

        cv::putText(frame,
                    "In zone : " + std::to_string(m_current_in_zone),
                    cv::Point(10, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.68,
                    cv::Scalar(0, 230, 0), 2);

        cv::putText(frame,
                    "Entries : " + std::to_string(m_total_entries),
                    cv::Point(10, 58),
                    cv::FONT_HERSHEY_SIMPLEX, 0.68,
                    cv::Scalar(0, 200, 255), 2);

        cv::putText(frame,
                    "Exits   : " + std::to_string(m_total_exits),
                    cv::Point(10, 88),
                    cv::FONT_HERSHEY_SIMPLEX, 0.68,
                    cv::Scalar(0, 120, 255), 2);
    }
}
