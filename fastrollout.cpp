#include <v4r/cuda.hpp>
#include <PathFinder.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <simdjson.h>
#include <zlib.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <fstream>
#include <filesystem>
#include <thread>
#include <future>
#include <queue>
#include <random>
#include <utility>
#include <string_view>
#include <vector>

using namespace std;
using namespace v4r;
namespace py = pybind11;

namespace SimulatorConfig {
    constexpr float SUCCESS_REWARD = 2.5;
    constexpr float SLACK_REWARD = 1e-2;
    constexpr float SUCCESS_DISTANCE = 0.2;
    constexpr float MAX_STEPS = 500;
}

struct Episode {
    glm::vec3 startPosition;
    glm::vec4 startRotation;
    glm::vec3 goal;
};

struct SceneMetadata {
    size_t firstEpisode;
    size_t numEpisodes;
    string scenePath;
};

static simdjson::dom::element parseFile(const string &file_path,
                                        size_t num_bytes,
                                        simdjson::dom::parser &parser)
{
    gzFile gz = gzopen(file_path.c_str(), "rb");
    if (gz == nullptr) {
        cerr << "Failed to open " << file_path << endl;
        abort();
    }

    vector<uint8_t> out_data {};

    size_t cur_out_size = num_bytes * 2;
    int cur_decompressed = 0;
    size_t total_decompressed = 0;
    for (int cur_decompressed = 0;
         !gzeof(gz) && cur_decompressed >= 0;
         cur_decompressed = gzread(gz, out_data.data() + total_decompressed,
                                   cur_out_size - total_decompressed),
         total_decompressed += cur_decompressed,
         cur_out_size *= 2) {

        out_data.resize(cur_out_size + simdjson::SIMDJSON_PADDING);
    }

    if (cur_decompressed == -1) {
        cerr << "Failed to read " << file_path << endl;
        abort();
    }

    gzclose(gz);

    return parser.parse(out_data.data(), total_decompressed, false);
}

template <typename T>
class Span {
public:
    Span(T *base, size_t num_elems)
        : ptr_(base), num_elems_(num_elems)
    {}

    T & operator[](size_t idx) { return ptr_[idx]; }
    const T & operator[](size_t idx) const { return ptr_[idx]; }

    T * begin() { return ptr_; }
    const T * begin() const { return ptr_; }

    T * end() { return ptr_ + num_elems_; }
    const T * end() const { return ptr_ + num_elems_; }

    size_t size() const { return num_elems_; }

private:
    T *ptr_;
    size_t num_elems_;
};

class Dataset {
public:
    Dataset(const string &dataset_path_name,
            const string &asset_path_name,
            uint32_t num_threads)
        : episodes_(),
          scenes_()
    {
        filesystem::path dataset_name {dataset_path_name};
        constexpr const char *data_suffix = ".json.gz";

        vector<pair<string, size_t>> json_files;
        for (const auto &entry : 
                filesystem::directory_iterator(dataset_name)) {

            const string filename = entry.path().string();
            if (string_view(filename).substr(
                    filename.size() - strlen(data_suffix)) == data_suffix) {
                json_files.push_back({
                    filename,
                    filesystem::file_size(entry.path())
                });
            }
        }

        num_threads =
            min(num_threads, static_cast<uint32_t>(json_files.size()));

        uint32_t files_per_thread = json_files.size() / num_threads;
        uint32_t extra_files =
            json_files.size() - num_threads * files_per_thread;

        vector<thread> loader_threads;
        loader_threads.reserve(num_threads);

        mutex merge_mutex;

        uint32_t thread_file_offset = 0;
        for (uint32_t i = 0; i < num_threads; i++) {
            uint32_t num_files = files_per_thread;
            if (extra_files > 0) {
                num_files++;
                extra_files--;
            }

            loader_threads.emplace_back(
                    [this, thread_file_offset, num_files,
                     &merge_mutex, &asset_path_name, &json_files]() {
                vector<Episode> episodes;
                vector<SceneMetadata> scenes;

                simdjson::dom::parser parser;

                for (uint32_t file_idx = 0; file_idx < num_files;
                     file_idx++) {
                    size_t scene_episode_start = episodes.size();
                    string_view scene_id;

                    const auto &[file_name, num_bytes] =
                        json_files[thread_file_offset + file_idx];

                    auto json = parseFile(file_name, num_bytes, parser);
                    const auto &json_episodes = json["episodes"];

                    for (const auto &json_episode : json_episodes) {
                        auto fill_vec = [](auto &vec, const auto &json_arr) {
                            size_t idx = 0;
                            for (double component : json_arr) {
                                vec[idx] = component;
                                idx++;
                            }
                        };

                        glm::vec3 start_pos;
                        fill_vec(start_pos, json_episode["start_position"]);
                        glm::vec4 start_rot;
                        fill_vec(start_rot, json_episode["start_rotation"]);
                        
                        glm::vec3 goal_pos;
                        fill_vec(goal_pos, json_episode["goals"].at(0)["position"]);

                        const string_view cur_scene_path = json_episode["scene_id"];

                        if (scene_id.size() == 0) {
                            scene_id = cur_scene_path;
                        }

                        if (scene_id != cur_scene_path) {
                            cerr << "Loading code assumes json file contains data for one scene" << endl;
                            abort();
                        }

                        episodes.push_back({
                            start_pos,
                            start_rot,
                            goal_pos
                        });
                    }

                    if (scene_id.size() > 0) {
                        scenes.push_back({
                            scene_episode_start,
                            episodes.size() - scene_episode_start,
                            asset_path_name + "/" + string(scene_id)
                        });
                    }
                }

                {
                    lock_guard merge_lock { merge_mutex };
                    uint32_t episode_offset = episodes_.size();

                    for (SceneMetadata &scene : scenes) {
                        scene.firstEpisode += episode_offset;
                        scenes_.push_back(scene);
                    }

                    for (const Episode &episode : episodes) {
                        episodes_.push_back(episode);
                    }
                }
            });

            thread_file_offset += num_files;
        }

        for (uint32_t i = 0; i < num_threads; i++) {
            loader_threads[i].join();
        }
    }

    Span<const Episode> getEpisodes(size_t scene_idx) const
    {
        const SceneMetadata &scene  = scenes_[scene_idx];
        return Span(&episodes_[scene.firstEpisode], scene.numEpisodes);
    }

    const string_view getScenePath(size_t scene_idx) const
    {
        return scenes_[scene_idx].scenePath;
    }

    size_t numScenes() const
    {
        return scenes_.size();
    }

private:
    vector<Episode> episodes_;
    vector<SceneMetadata> scenes_;
};

BatchRendererCUDA makeRenderer(int32_t gpu_id, uint32_t renderer_batch_size,
                               const array<uint32_t, 2> &resolution,
                               bool color, bool depth, bool doubleBuffered) {
    RenderOptions options {};
    if (doubleBuffered) {
        options |= RenderOptions::DoubleBuffered;
    }

    auto make = [&](auto features) {
        return BatchRendererCUDA(
            {gpu_id, // gpuID
             1,      // numLoaders
             1,      // numStreams
             renderer_batch_size, resolution[1], resolution[0],
             glm::mat4(1, 0, 0, 0,
                       0, -1.19209e-07, -1, 0,
                       0, 1, -1.19209e-07, 0,
                       0, 0, 0, 1) // Habitat coordinate txfm matrix
             ,
             }, features);
    };

    if (color && depth) {
        return make(RenderFeatures<Unlit<RenderOutputs::Color | RenderOutputs::Depth,
                                   DataSource::Texture>> { options });
    } else if (color) {
        return make(RenderFeatures<Unlit<RenderOutputs::Color,
                                   DataSource::Texture>> { options });
    } else {
        return make(RenderFeatures<Unlit<RenderOutputs::Depth,
                                   DataSource::None>> { options });
    }

}

template <typename R> bool isReady(const future<R> &f) {
    return f.wait_for(chrono::seconds(0)) == future_status::ready;
}

struct BackgroundSceneLoader {
    explicit BackgroundSceneLoader(AssetLoader &&loader)
        : loader_{move(loader)}, loader_mutex_{}, loader_cv_{},
          loader_exit_{false}, loader_requests_{}, loader_thread_{[&]() {
              loaderLoop();
          }} {};

    ~BackgroundSceneLoader() {
        {
            lock_guard<mutex> cv_lock(loader_mutex_);
            loader_exit_ = true;
        }
        loader_cv_.notify_one();
        loader_thread_.join();
    }

    shared_ptr<Scene> loadScene(const string &scene_path) {
        auto fut = asyncLoadScene(scene_path);
        fut.wait();
        return fut.get();
    }

    future<shared_ptr<Scene>> asyncLoadScene(const string &scene_path) {
        future<shared_ptr<Scene>> loader_future;

        {
            lock_guard<mutex> wait_lock(loader_mutex_);

            promise<shared_ptr<Scene>> loader_promise;
            loader_future = loader_promise.get_future();

            loader_requests_.emplace(scene_path, move(loader_promise));
        }
        loader_cv_.notify_one();

        return loader_future;
    }

  private:
    void loaderLoop() {
        while (true) {
            string scene_path;
            promise<shared_ptr<Scene>> loader_promise;
            {
                unique_lock<mutex> wait_lock(loader_mutex_);
                while (loader_requests_.size() == 0) {
                    loader_cv_.wait(wait_lock);
                    if (loader_exit_) {
                        return;
                    }
                }

                scene_path = move(loader_requests_.front().first);
                loader_promise = move(loader_requests_.front().second);
                loader_requests_.pop();
            }

            auto scene = loader_.loadScene(scene_path);
            loader_promise.set_value(move(scene));
        }
    };

    AssetLoader loader_;

    mutex loader_mutex_;
    condition_variable loader_cv_;
    bool loader_exit_;
    queue<pair<string, promise<shared_ptr<Scene>>>> loader_requests_;
    thread loader_thread_;
};

class Renderer {
public:
    Renderer(int gpu_id, uint32_t batch_size,
             const array<uint32_t, 2> &resolution, bool color,
             bool depth, uint32_t num_groups)
        : renderer_{makeRenderer(gpu_id, batch_size,
                                 resolution, color, depth, num_groups == 2)},
          loader_{renderer_.makeLoader()},
          cmd_strm_{renderer_.makeCommandStream()}
    {}

    ~Renderer() = default;

    uint8_t * getColorMemory(const uint32_t groupIdx) const {
        return cmd_strm_.getColorDevicePtr(groupIdx);
    }

    float * getDepthMemory(const uint32_t groupIdx) const {
        return cmd_strm_.getDepthDevicePtr(groupIdx);
    }

    bool isLoadingNextScene() {
        return nextScene_ != nullptr || newSceneFuture_.valid();
    }

    bool nextSceneLoaded() {
        return newSceneFuture_.valid() && isReady(newSceneFuture_);
    }

    void loadNewScene(const string &scenePath) {
        assert(!isLoadingNextScene() && "Already loading the next scene!");
        newSceneFuture_ = loader_.asyncLoadScene(scenePath);
    }

    bool hasNextScene() { return nextScene_ != nullptr; }

    void acquireNextScene() {
        newSceneFuture_.wait();
        nextScene_ = newSceneFuture_.get();
    }

    void doneSwappingScene() { nextScene_ = nullptr; }

    shared_ptr<Scene> getNextScene() {
        assert(hasNextScene() && "Next scene is nullptr");

        return nextScene_;
    }

private:
    BatchRendererCUDA renderer_;
    BackgroundSceneLoader loader_;
    CommandStreamCUDA cmd_strm_;

    future<shared_ptr<Scene>> newSceneFuture_;
    shared_ptr<Scene> nextScene_ = nullptr;
};

#if 0
        for (uint32_t env_idx = 0; env_idx < envs_.size(); env_idx++) {
            envs_[env_idx].setCameraView(views[env_idx]);
        }
             const vector<string> &initial_scene_paths,
             uint32_t batch_size_per_scene, int gpu_id,

                                 batch_size_per_scene *
                                 static_cast<uint32_t>(
                                     initial_scene_paths.size() / num_groups),

        for (uint32_t i = 0, sceneIdx = 0; i < num_groups; ++i) {
            vector<Environment> envs;
            for (uint32_t __i = 0; __i < scene_paths.size() / num_groups;
                 ++__i, ++sceneIdx) {

                auto scene = loader_.loadScene(scene_paths[sceneIdx]);
                for (uint32_t __j = 0; __j < batch_size_per_scene; ++__j)
                    envs.emplace_back(
                        move(cmd_strm_.makeEnvironment(scene, 90, 0.01, 1000)));
            }

            renderGroups_.emplace_back(new EnvironmentGroup(
                cmd_strm_, move(envs), batch_size_per_scene,
                cmd_strm_.getColorDevicePtr(i),
                cmd_strm_.getDepthDevicePtr(i)));
        }

        envsPerGroup_{numScenes_ * batch_size_per_scene_ / num_groups},
#endif

class SimulatorState {
    glm::vec3 curPosition;
};

struct EnvironmentGroup {
    EnvironmentGroup(CommandStreamCUDA &strm,
                     AssetLoader &loader,
                     const Dataset &dataset,
                     size_t envs_per_scene,
                     const Span<int> &initial_scene_indices)
        : cmd_strm_{strm},
          sim_states_(),
          render_envs_()
    {
        sim_states_.reserve(envs_per_scene * initial_scene_indices.size());
        render_envs_.reserve(envs_per_scene * initial_scene_indices.size());

        for (int scene_idx : initial_scene_indices) {
            auto scene_path = dataset.getScenePath(scene_idx);
            auto scene = loader.loadScene(scene_path);
            for (size_t env_idx = 0; env_idx < envs_per_scene; env_idx++) {
                render_envs_.emplace_back(
                        strm.makeEnvironment(scene, 90, 0.1, 1000));
            }
        }
    }

    void render()
    {
        cmd_strm_.render(render_envs_);
    }

    Environment &getEnvironment(size_t idx)
    {
        return render_envs_[idx];
    }

    void swapScene(const uint32_t envIdx, const shared_ptr<Scene> &nextScene) {
        render_envs_[envIdx] = cmd_strm_.makeEnvironment(nextScene, 90, 0.01, 1000);
    }

  private:
    CommandStreamCUDA &cmd_strm_;
    vector<SimulatorState> sim_states_;
    vector<Environment> render_envs_;
};

class RolloutGenerator {
public:
    RolloutGenerator(const string &dataset_path, const string &asset_path,
                     uint32_t num_environments, uint32_t num_active_scenes,
                     int num_workers, int gpu_id,
                     const array<uint32_t, 2> &render_resolution,
                     bool color, bool depth, bool double_buffered)
        : RolloutGenerator(dataset_path, asset_path,
                           num_environments, num_active_scenes,
                           num_workers == -1 ?
                               min(static_cast<int64_t>(
                                    thread::hardware_concurrency()) - 1, 1l) :
                               num_workers,
                           gpu_id, render_resolution, color, depth,
                           double_buffered ? 2u : 1u)
    {}

    void step(size_t group_idx, const int64_t *action_ptr)
    {
        EnvironmentGroup &cur_group = groups_[group_idx];

        cur_group.render();
    }

private:
    RolloutGenerator(const string &dataset_path, const string &asset_path,
                    uint32_t num_environments, uint32_t num_active_scenes,
                    int num_workers, int gpu_id,
                    const array<uint32_t, 2> &render_resolution,
                    bool color, bool depth, uint32_t num_groups)
        : dataset_(dataset_path, asset_path, num_workers),
          renderer_(makeRenderer(gpu_id, num_environments / num_groups,
                                 render_resolution,
                                 color, depth, num_groups == 2)),
          cmd_strm_(renderer_.makeCommandStream()),
          loader_(renderer_.makeLoader()),
          rd_(),
          rgen_(rd_()),
          inactive_scenes_(),
          groups_(),
          worker_threads_()
    {
        groups_.reserve(num_groups);
        worker_threads_.reserve(num_workers);

        vector<int> initial_scene_idxs;
        initial_scene_idxs.reserve(num_active_scenes);
        inactive_scenes_.reserve(dataset_.numScenes() - num_active_scenes);

        uniform_real_distribution<> selection_distribution(0.f, 1.f);
        for (int i = 0 ; i < dataset_.numScenes() &&
             initial_scene_idxs.size() < num_active_scenes; i++) {
            float weight = selection_distribution(rgen_);
            if ((dataset_.numScenes() - i) * weight >=
                (num_active_scenes - initial_scene_idxs.size())) {
                initial_scene_idxs.push_back(i);
            } else {
                inactive_scenes_.push_back(i);
            }
        }

        assert(num_environments % num_groups == 0);
        assert(num_environments % num_active_scenes == 0);
        assert(num_active_scenes % num_groups == 0);

        uint32_t scenes_per_group = num_active_scenes / num_groups;
        uint32_t envs_per_scene = num_environments / num_active_scenes;

        for (int i = 0; i < num_groups; i++) {
            groups_.emplace_back(cmd_strm_, loader_, dataset_,
                envs_per_scene, Span(&initial_scene_idxs[i * scenes_per_group],
                                     scenes_per_group));
        }

        for (int thread_idx = 0; thread_idx < num_workers; thread_idx++) {
            worker_threads_.emplace_back([&, thread_idx]() {
            });
        }
    }

    Dataset dataset_;
    BatchRendererCUDA renderer_;
    CommandStreamCUDA cmd_strm_;
    AssetLoader loader_;
    random_device rd_;
    mt19937 rgen_;
    vector<int> inactive_scenes_;
    vector<EnvironmentGroup> groups_;
    vector<thread> worker_threads_;
};

PYBIND11_MODULE(ddppo_fastrollout, m) {
    py::class_<RolloutGenerator>(m, "RolloutGenerator")
        .def(py::init<const string &, const string &,
                      uint32_t, uint32_t, int, int,
                      const array<uint32_t, 2> &,
                      bool, bool, bool>())
        .def("step", &RolloutGenerator::step);
        //.def("rgba", &RolloutGenerator::getColorMemory)
        //.def("depth", &RolloutGenerator::getDepthMemory)
        //.def("done_swapping_scenes", &DoubleBuffered::doneSwappingScene)
        //.def("acquire_next_scene", &DoubleBuffered::acquireNextScene)
        //.def("load_new_scene", &DoubleBuffered::loadNewScene)
        //.def("swap_scene", &DoubleBuffered::swapScene)
        //.def("next_scene_loaded", &DoubleBuffered::nextSceneLoaded)
        //.def("has_next_scene", &DoubleBuffered::hasNextScene)
        //.def("is_loading_next_scene", &DoubleBuffered::isLoadingNextScene);
}
