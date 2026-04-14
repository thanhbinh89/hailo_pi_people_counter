/**
 * people_counter.cpp
 *
 * Real-time people counting in a configurable zone using Hailo inference.
 *
 * Features
 * --------
 *  • Counts people currently inside a configurable rectangular zone
 *  • Tracks cumulative entry and exit events
 *  • Tracks per-person dwell time (seconds) while inside the zone
 *  • All zone geometry configured via config.yaml (normalized 0-1 coordinates)
 */

#include "toolbox.hpp"
#include "hailo_infer.hpp"
#include "zone_tracker.hpp"

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <future>
#include <algorithm>
#include <unistd.h>

using namespace hailo_utils;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

// Required by toolbox.hpp (extern declaration)
std::vector<cv::Scalar> COLORS = {
    cv::Scalar(255,   0,   0),  // Red
    cv::Scalar(  0, 255,   0),  // Green
    cv::Scalar(  0,   0, 255),  // Blue
    cv::Scalar(255, 255,   0),  // Cyan
    cv::Scalar(255,   0, 255),  // Magenta
    cv::Scalar(  0, 255, 255),  // Yellow
    cv::Scalar(255, 128,   0),  // Orange
    cv::Scalar(128,   0, 128),  // Purple
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

constexpr size_t MAX_QUEUE_SIZE    = 30;
constexpr size_t COCO_CLASS_PERSON = 0;   // COCO index 0 = person
constexpr size_t COCO_NUM_CLASSES  = 80;

// ─────────────────────────────────────────────────────────────────────────────
// Shared queues (pipeline: preprocess → inference → postprocess)
// ─────────────────────────────────────────────────────────────────────────────

static std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>
    g_preprocessed_queue = std::make_shared<
        BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>>(MAX_QUEUE_SIZE);

static std::shared_ptr<BoundedTSQueue<InferenceResult>>
    g_results_queue = std::make_shared<BoundedTSQueue<InferenceResult>>(MAX_QUEUE_SIZE);

// ─────────────────────────────────────────────────────────────────────────────
// Preprocess callback: BGR frame → RGB resized for model
// ─────────────────────────────────────────────────────────────────────────────

static void preprocess_cb(const std::vector<cv::Mat>& org_frames,
                           std::vector<cv::Mat>&       out_frames,
                           uint32_t target_w, uint32_t target_h)
{
    out_frames.clear();
    out_frames.reserve(org_frames.size());

    for (const auto& src : org_frames) {
        if (src.empty()) { out_frames.emplace_back(); continue; }

        cv::Mat rgb;
        if      (src.channels() == 3) cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
        else if (src.channels() == 4) cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB);
        else                          cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);

        if (rgb.cols != static_cast<int>(target_w) ||
            rgb.rows != static_cast<int>(target_h))
        {
            cv::resize(rgb, rgb,
                       cv::Size(static_cast<int>(target_w),
                                static_cast<int>(target_h)),
                       0, 0, cv::INTER_AREA);
        }
        if (!rgb.isContinuous()) rgb = rgb.clone();
        out_frames.push_back(std::move(rgb));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parse NMS output — person detections only
//
// Hailo NMS layout:
//   for each class c in [0, 80):
//     float32  count
//     count ×  hailo_bbox_float32_t  { x_min, y_min, x_max, y_max, score }
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<cv::Rect> parse_person_detections(
    uint8_t* data, int frame_w, int frame_h, float score_thresh)
{
    std::vector<cv::Rect> rects;
    size_t offset = 0;

    for (size_t cls = 0; cls < COCO_NUM_CLASSES; ++cls) {
        auto count = static_cast<uint32_t>(
            *reinterpret_cast<float32_t*>(data + offset));
        offset += sizeof(float32_t);

        for (uint32_t j = 0; j < count; ++j) {
            hailo_bbox_float32_t bbox =
                *reinterpret_cast<hailo_bbox_float32_t*>(data + offset);
            offset += sizeof(hailo_bbox_float32_t);

            if (cls == COCO_CLASS_PERSON && bbox.score >= score_thresh) {
                int x1 = std::max(0, static_cast<int>(bbox.x_min * frame_w));
                int y1 = std::max(0, static_cast<int>(bbox.y_min * frame_h));
                int x2 = std::min(frame_w - 1, static_cast<int>(bbox.x_max * frame_w));
                int y2 = std::min(frame_h - 1, static_cast<int>(bbox.y_max * frame_h));

                if (x2 > x1 && y2 > y1)
                    rects.emplace_back(cv::Point(x1, y1), cv::Point(x2, y2));
            }
        }
    }
    return rects;
}

// ─────────────────────────────────────────────────────────────────────────────
// Postprocess callback factory
// ─────────────────────────────────────────────────────────────────────────────

static PostprocessCallback make_postprocess_cb(
    std::shared_ptr<ZoneTracker> tracker,
    AppConfig cfg)
{
    return [tracker, cfg](
        cv::Mat& frame,
        const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>>& outputs)
    {
        auto person_rects = parse_person_detections(
            outputs[0].first, frame.cols, frame.rows, cfg.confidence_threshold);

        // Draw bounding boxes (orange)
        for (const auto& r : person_rects)
            cv::rectangle(frame, r, cv::Scalar(0, 165, 255), 2);

        tracker->update(person_rects, frame.cols, frame.rows);
        tracker->draw(frame);
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    try {
        const std::string APP_NAME = "people_counter";

        // ── Config path ───────────────────────────────────────────────────────
        std::string config_path;
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--config") {
                config_path = argv[i + 1];
                break;
            }
        }
        if (config_path.empty()) {
            char exe_buf[4096] = {};
            ssize_t len = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
            config_path = (len > 0)
                ? (fs::path(exe_buf).parent_path() / "config.yaml").string()
                : "config.yaml";
        }

        std::cout << "Loading config: " << config_path << "\n";
        AppConfig cfg = load_config(config_path);

        std::cout << "Zone       : (" << cfg.zone_x1 << ", " << cfg.zone_y1
                  << ") – (" << cfg.zone_x2 << ", " << cfg.zone_y2 << ") [normalized]\n";
        std::cout << "Confidence : " << cfg.confidence_threshold << "\n";
        std::cout << "Max dist   : " << cfg.max_distance << " px\n";
        std::cout << "Max disapp : " << cfg.max_disappeared << " frames\n\n";

        // ── Hailo CLI args ────────────────────────────────────────────────────
        CommandLineArgs args = parse_command_line_arguments(argc, argv);
        post_parse_args(APP_NAME, args, argc, argv);

        // ── Tracker & model ───────────────────────────────────────────────────
        auto tracker = std::make_shared<ZoneTracker>(cfg);
        HailoInfer   model(args.net, args.batch_size);

        // ── Open input ────────────────────────────────────────────────────────
        double           org_h = 0, org_w = 0;
        size_t           frame_count = 0;
        cv::VideoCapture capture;
        InputType        input_type;

        input_type = determine_input_type(
            args.input, std::ref(capture),
            std::ref(org_h), std::ref(org_w), std::ref(frame_count),
            std::ref(args.batch_size), std::ref(args.camera_resolution));

        // ── Postprocess callback ──────────────────────────────────────────────
        auto post_cb = make_postprocess_cb(tracker, cfg);

        // ── Launch 3-thread pipeline ──────────────────────────────────────────
        auto t_start = Clock::now();
        std::chrono::duration<double> inference_time;

        auto preprocess_thread = std::async(run_preprocess,
            std::ref(args.input),
            std::ref(args.net),
            std::ref(model),
            std::ref(input_type),
            std::ref(capture),
            std::ref(args.batch_size),
            std::ref(args.framerate),
            g_preprocessed_queue,
            preprocess_cb);

        ModelInputQueuesMap input_queues = {
            { model.get_infer_model()->get_input_names().at(0), g_preprocessed_queue }
        };

        auto inference_thread = std::async(run_inference_async,
            std::ref(model),
            std::ref(inference_time),
            std::ref(input_queues),
            g_results_queue);

        auto postprocess_thread = std::async(run_post_process,
            std::ref(input_type),
            std::ref(org_h),
            std::ref(org_w),
            std::ref(frame_count),
            std::ref(capture),
            std::ref(args.framerate),
            std::ref(args.batch_size),
            std::ref(args.save_stream_output),
            std::ref(args.no_display),
            std::ref(args.output_dir),
            std::ref(args.output_resolution),
            g_results_queue,
            post_cb);

        // ── Wait ──────────────────────────────────────────────────────────────
        hailo_status status = wait_and_check_threads(
            preprocess_thread,  "Preprocess",
            inference_thread,   "Inference",
            postprocess_thread, "Postprocess");

        if (HAILO_SUCCESS != status) return static_cast<int>(status);

        auto t_end = Clock::now();
        print_inference_statistics(inference_time, args.net,
            static_cast<double>(frame_count), t_end - t_start);

        // ── Final summary ─────────────────────────────────────────────────────
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════╗\n";
        std::cout << "║      People Counter — Summary        ║\n";
        std::cout << "╠══════════════════════════════════════╣\n";
        std::cout << "║  Total entries : "
                  << std::setw(6) << tracker->total_entries()
                  << "                ║\n";
        std::cout << "║  Total exits   : "
                  << std::setw(6) << tracker->total_exits()
                  << "                ║\n";
        std::cout << "║  Still in zone : "
                  << std::setw(6) << tracker->current_in_zone()
                  << "                ║\n";
        std::cout << "╚══════════════════════════════════════╝\n\n";

        return HAILO_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return HAILO_INTERNAL_FAILURE;
    }
}
