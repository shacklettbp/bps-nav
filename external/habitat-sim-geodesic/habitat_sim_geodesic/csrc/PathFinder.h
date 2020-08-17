// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Eigen>

#include "spimpl.h"

namespace esp {
typedef Eigen::Vector3f vec3f;

// smart pointers macro
#define ESP_SMART_POINTERS(T)                                 \
 public:                                                      \
  typedef std::shared_ptr<T> ptr;                             \
  typedef std::unique_ptr<T> uptr;                            \
  typedef std::shared_ptr<const T> cptr;                      \
  typedef std::unique_ptr<const T> ucptr;                     \
  template <typename... Targs>                                \
  static inline ptr create(Targs&&... args) {                 \
    return std::make_shared<T>(std::forward<Targs>(args)...); \
  }                                                           \
  template <typename... Targs>                                \
  static inline uptr create_unique(Targs&&... args) {         \
    return std::make_unique<T>(std::forward<Targs>(args)...); \
  }

// pimpl macro backed by unique_ptr pointer
#define ESP_UNIQUE_PTR_PIMPL() \
 protected:                    \
  struct Impl;                 \
  spimpl::unique_impl_ptr<Impl> pimpl_;

// pimpl macro backed by shared_ptr pointer
#define ESP_SHARED_PTR_PIMPL() \
 protected:                    \
  struct Impl;                 \
  spimpl::impl_ptr<Impl> pimpl_;

// convenience macros with combined smart pointers and pimpl members
#define ESP_SMART_POINTERS_WITH_UNIQUE_PIMPL(T) \
  ESP_SMART_POINTERS(T)                         \
  ESP_UNIQUE_PTR_PIMPL()
#define ESP_SMART_POINTERS_WITH_SHARED_PIMPL(T) \
  ESP_SMART_POINTERS(T)                         \
  ESP_SHARED_PTR_PIMPL()

namespace nav {

struct NavMeshPoint {
  vec3f xyz;
  uint64_t polyId;
};

/**
 * @brief Struct for shortest path finding. Used in conjunction with @ref
 * PathFinder.findPath
 */
struct ShortestPath {
  /**
   * @brief The starting point for the path
   */
  NavMeshPoint requestedStart;

  /**
   * @brief The ending point for the path
   */
  NavMeshPoint requestedEnd;

  /**
   * @brief A list of points that specify the shortest path on the navigation
   * mesh between @ref requestedStart and @ref requestedEnd
   *
   * @note Will be empty if no path exists
   */
  std::vector<vec3f> points;

  /**
   * @brief The geodesic distance between @ref requestedStart and @ref
   * requestedEnd
   *
   * @note Will be inf if no path exists
   */
  float geodesicDistance;

  ESP_SMART_POINTERS(ShortestPath)
};

/** Loads and/or builds a navigation mesh and then performs path
 * finding and collision queries on that navmesh
 *
 */
class PathFinder {
 public:
  /**
   * @brief Constructor.
   */
  PathFinder();
  ~PathFinder() = default;

  /**
   * @brief Returns a random navigable point
   *
   * @return A random navigable point.
   *
   * @note This method can fail.  If it does,
   * the returned point will be arbitrary and may not be navigable. Use @ref
   * isNavigable to check if the point is navigable.
   */
  vec3f getRandomNavigablePoint();

  /**
   * @brief Finds the shortest path between two points on the navigation mesh
   *
   * @param[inout] path The @ref ShortestPath structure contain the starting
   * and end point. This method will populate the @ref ShortestPath.points and
   * @ref ShortestPath.geodesicDistance fields.
   *
   * @return Whether or not a path exists between @ref
   * ShortestPath.requestedStart and @ref ShortestPath.requestedEnd
   */
  bool findPath(ShortestPath& path);

  /**
   * @brief Attempts to move from @ref start to @ref end and returns the
   * navigable point closest to @ref end that is feasibly reachable from @ref
   * start
   *
   * @param[in] start The starting location
   * @param[out] end The desired end location
   *
   * @return The found end location
   */
  NavMeshPoint tryStep(const NavMeshPoint& start, const esp::vec3f& end);

  /**
   * @brief Same as @ref tryStep but does not allow for sliding along walls
   */
  NavMeshPoint tryStepNoSliding(const NavMeshPoint& start,
                                const esp::vec3f& end);

  /**
   * @brief Snaps a point to the navigation mesh
   *
   * @param[in] pt The point to snap to the navigation mesh
   *
   * @return The closest navigation point to @ref pt.  Will be {inf, inf, inf}
   * if no navigable point was within a reasonable distance
   */
  NavMeshPoint snapPoint(const esp::vec3f& pt);

  /**
   * @brief Loads a navigation meshed saved by @ref saveNavMesh
   *
   * @param[in] path The saved navigation mesh file, generally has extension
   * ``.navmesh``
   *
   * @return Whether or not the navmesh was successfully loaded
   */
  bool loadNavMesh(const std::string& path);

  /**
   * @brief Saves a navigation mesh to later be loaded by @ref loadNavMesh
   *
   * @param[in] path The name of the file, generally has extension
   * ``.navmesh``
   *
   * @return Whether or not the navmesh was successfully saved
   */
  bool saveNavMesh(const std::string& path);

  /**
   * @return If a navigation mesh is current loaded or not
   */
  bool isLoaded() const;

  /**
   * @brief Seed the pathfinder.  Useful for @ref getRandomNavigablePoint
   *
   * @param[in] newSeed The random seed
   *
   * @note This just seeds the global c @ref rand function.
   */
  void seed(uint32_t newSeed);

  /**
   * @brief returns the size of the connected component @ ref pt belongs to.
   *
   * @param[in] pt The point to specify the connected component
   *
   * @return Size of the connected component
   */
  float islandRadius(const vec3f& pt) const;

  /**
   * @brief Query whether or not a given location is navigable
   *
   * This method works by snapping @ref pt to the navigation mesh with @ref
   * snapPoint and then checking to see if there was no displacement in the
   * x-z plane and at most @ref maxYDelta displacement in the y direction.
   *
   * @param[in] pt The location to check whether or not it is navigable
   * @param[in] maxYDelta The maximum y displacement.  This tolerance is
   * useful for computing a top-down occupancy grid as the floor is not
   * perfectly level
   *
   * @return Whether or not @ref pt is navigable
   */
  bool isNavigable(const vec3f& pt, const float maxYDelta = 0.5) const;

  /**
   * @return The axis aligned bounding box containing the navigation mesh.
   */
  std::pair<vec3f, vec3f> bounds() const;

  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> getTopDownView(
      const float pixelsPerMeter,
      const float height);

  ESP_SMART_POINTERS_WITH_UNIQUE_PIMPL(PathFinder);
};

}  // namespace nav
}  // namespace esp
