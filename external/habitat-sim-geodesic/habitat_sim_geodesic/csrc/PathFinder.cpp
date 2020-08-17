// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "PathFinder.h"
#include <stack>
#include <unordered_map>

#include <cstdio>
#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>

#include "DetourNavMesh.h"
#include "DetourNavMeshBuilder.h"
#include "DetourNavMeshQuery.h"
#include "DetourNode.h"

namespace esp {
namespace nav {

namespace {
template <typename T>
std::tuple<dtStatus, dtPolyRef, vec3f> projectToPoly(
    const T& pt,
    const dtNavMeshQuery* navQuery,
    const dtQueryFilter* filter) {
  // Defines size of the bounding box to search in for the nearest polygon. If
  // there is no polygon inside the bounding box, the status is set to failure
  // and polyRef == 0
  constexpr float polyPickExt[3] = {2, 4, 2};  // [2 * dx, 2 * dy, 2 * dz]
  dtPolyRef polyRef;
  // Initialize with all NANs at dtStatusSucceed(status) == true does NOT mean
  // that it found a point to project to..........
  vec3f polyXYZ{NAN, NAN, NAN};
  dtStatus status = navQuery->findNearestPoly(pt.data(), polyPickExt, filter,
                                              &polyRef, polyXYZ.data());

  // So let's call it a failure if it didn't actually find a point....
  if (std::isnan(polyXYZ[0]))
    status = DT_FAILURE;

  return std::make_tuple(status, polyRef, polyXYZ);
}
}  // namespace

namespace impl {

// Runs connected component analysis on the navmesh to figure out which polygons
// are connected This gives O(1) lookup for if a path between two polygons
// exists or not
// Takes O(npolys) to construct
class IslandSystem {
 public:
  IslandSystem(const dtNavMesh* navMesh, const dtQueryFilter* filter) {
    std::vector<vec3f> islandVerts;

    // Iterate over all tiles
    for (int iTile = 0; iTile < navMesh->getMaxTiles(); ++iTile) {
      const dtMeshTile* tile = navMesh->getTile(iTile);
      if (!tile)
        continue;

      // Iterate over all polygons in a tile
      for (int jPoly = 0; jPoly < tile->header->polyCount; ++jPoly) {
        // Get the polygon reference from the tile and polygon id
        dtPolyRef startRef = navMesh->encodePolyId(iTile, tile->salt, jPoly);

        // If the polygon ref is valid, and we haven't seen it yet,
        // start connected component analysis from this polygon
        if (navMesh->isValidPolyRef(startRef) &&
            (polyToIsland_.find(startRef) == polyToIsland_.end())) {
          uint32_t newIslandId = islandRadius_.size();
          expandFrom(navMesh, filter, newIslandId, startRef, islandVerts);

          // The radius is calculated as the max deviation from the
          // mean for all points in the island
          vec3f centroid = vec3f::Zero();
          for (auto& v : islandVerts) {
            centroid += v;
          }
          centroid /= islandVerts.size();

          float maxRadius = 0.0;
          for (auto& v : islandVerts) {
            maxRadius = std::max(maxRadius, (v - centroid).norm());
          }

          islandRadius_.emplace_back(maxRadius);
        }
      }
    }
  }

  inline bool hasConnection(dtPolyRef startRef, dtPolyRef endRef) const {
    // If both polygons are on the same island, there must be a path between
    // them
    auto itStart = polyToIsland_.find(startRef);
    if (itStart == polyToIsland_.end())
      return false;

    auto itEnd = polyToIsland_.find(endRef);
    if (itEnd == polyToIsland_.end())
      return false;

    return itStart->second == itEnd->second;
  }

  inline float islandRadius(dtPolyRef ref) const {
    auto itRef = polyToIsland_.find(ref);
    if (itRef == polyToIsland_.end())
      return 0.0;

    return islandRadius_[itRef->second];
  }

 private:
  std::unordered_map<dtPolyRef, uint32_t> polyToIsland_;
  std::vector<float> islandRadius_;

  void expandFrom(const dtNavMesh* navMesh,
                  const dtQueryFilter* filter,
                  const uint32_t newIslandId,
                  const dtPolyRef& startRef,
                  std::vector<vec3f>& islandVerts) {
    polyToIsland_.emplace(startRef, newIslandId);
    islandVerts.clear();

    // Force std::stack to be implemented via an std::vector as linked
    // lists are gross
    std::stack<dtPolyRef, std::vector<dtPolyRef>> stack;

    // Add the start ref to the stack
    stack.push(startRef);
    while (!stack.empty()) {
      dtPolyRef ref = stack.top();
      stack.pop();

      const dtMeshTile* tile = 0;
      const dtPoly* poly = 0;
      navMesh->getTileAndPolyByRefUnsafe(ref, &tile, &poly);

      for (int iVert = 0; iVert < poly->vertCount; ++iVert) {
        islandVerts.emplace_back(
            Eigen::Map<vec3f>(&tile->verts[poly->verts[iVert] * 3]));
      }

      // Iterate over all neighbours
      for (unsigned int iLink = poly->firstLink; iLink != DT_NULL_LINK;
           iLink = tile->links[iLink].next) {
        dtPolyRef neighbourRef = tile->links[iLink].ref;
        // If we've already visited this poly, skip it!
        if (polyToIsland_.find(neighbourRef) != polyToIsland_.end())
          continue;

        const dtMeshTile* neighbourTile = 0;
        const dtPoly* neighbourPoly = 0;
        navMesh->getTileAndPolyByRefUnsafe(neighbourRef, &neighbourTile,
                                           &neighbourPoly);

        // If a neighbour isn't walkable, don't add it
        if (!filter->passFilter(neighbourRef, neighbourTile, neighbourPoly))
          continue;

        polyToIsland_.emplace(neighbourRef, newIslandId);
        stack.push(neighbourRef);
      }
    }
  }
};
}  // namespace impl

struct PathFinder::Impl {
  Impl();
  ~Impl() = default;

  vec3f getRandomNavigablePoint();

  bool findPath(ShortestPath& path);

  NavMeshPoint tryStep(const NavMeshPoint& start,
                       const esp::vec3f& end,
                       bool allowSliding);

  NavMeshPoint snapPoint(const esp::vec3f& pt);

  bool loadNavMesh(const std::string& path);

  bool saveNavMesh(const std::string& path);

  bool isLoaded() const { return navMesh_ != nullptr; };

  void seed(uint32_t newSeed);

  float islandRadius(const vec3f& pt) const;

  bool isNavigable(const vec3f& pt, const float maxYDelta = 0.5) const;

  std::pair<vec3f, vec3f> bounds() const { return bounds_; };

  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> getTopDownView(
      const float pixelsPerMeter,
      const float height);

 private:
  struct NavMeshDeleter {
    void operator()(dtNavMesh* mesh) { dtFreeNavMesh(mesh); }
  };
  struct NavQueryDeleter {
    void operator()(dtNavMeshQuery* query) { dtFreeNavMeshQuery(query); }
  };

  std::unique_ptr<dtNavMesh, NavMeshDeleter> navMesh_ = nullptr;
  std::unique_ptr<dtNavMeshQuery, NavQueryDeleter> navQuery_ = nullptr;
  std::unique_ptr<dtQueryFilter> filter_ = nullptr;
  std::unique_ptr<impl::IslandSystem> islandSystem_ = nullptr;

  std::pair<vec3f, vec3f> bounds_;

  void removeZeroAreaPolys();
  bool initNavQuery();

  std::tuple<float, std::vector<vec3f>> findPathInternal(
      const NavMeshPoint& start,
      const NavMeshPoint& end);
};

namespace {
enum PolyAreas { POLYAREA_GROUND, POLYAREA_DOOR };

enum PolyFlags {
  POLYFLAGS_WALK = 0x01,      // walkable
  POLYFLAGS_DOOR = 0x02,      // ability to move through doors
  POLYFLAGS_DISABLED = 0x04,  // disabled polygon
  POLYFLAGS_ALL = 0xffff      // all abilities
};
}  // namespace

PathFinder::Impl::Impl() {
  filter_ = std::make_unique<dtQueryFilter>();
  filter_->setIncludeFlags(POLYFLAGS_WALK);
  filter_->setExcludeFlags(0);
}

bool PathFinder::Impl::initNavQuery() {
  navQuery_.reset(dtAllocNavMeshQuery());
  dtStatus status = navQuery_->init(navMesh_.get(), 2048);
  if (dtStatusFailed(status)) {
    return false;
  }

  islandSystem_ =
      std::make_unique<impl::IslandSystem>(navMesh_.get(), filter_.get());

  return true;
}

namespace {
const int NAVMESHSET_MAGIC = 'M' << 24 | 'S' << 16 | 'E' << 8 | 'T';  //'MSET';
const int NAVMESHSET_VERSION = 1;

struct NavMeshSetHeader {
  int magic;
  int version;
  int numTiles;
  dtNavMeshParams params;
};

struct NavMeshTileHeader {
  dtTileRef tileRef;
  int dataSize;
};

// Calculate the area of a polygon by iterating over the triangles in the detail
// mesh and computing their area
float polyArea(const dtPoly* poly, const dtMeshTile* tile) {
  float area = 0;
  // Code to iterate over triangles from here:
  // https://github.com/recastnavigation/recastnavigation/blob/57610fa6ef31b39020231906f8c5d40eaa8294ae/Detour/Source/DetourNavMesh.cpp#L684
  const std::ptrdiff_t ip = poly - tile->polys;
  const dtPolyDetail* pd = &tile->detailMeshes[ip];
  for (int j = 0; j < pd->triCount; ++j) {
    const unsigned char* t = &tile->detailTris[(pd->triBase + j) * 4];
    const float* v[3];
    for (int k = 0; k < 3; ++k) {
      if (t[k] < poly->vertCount)
        v[k] = &tile->verts[poly->verts[t[k]] * 3];
      else
        v[k] =
            &tile->detailVerts[(pd->vertBase + (t[k] - poly->vertCount)) * 3];
    }

    const vec3f w1 =
        Eigen::Map<const vec3f>(v[1]) - Eigen::Map<const vec3f>(v[0]);
    const vec3f w2 =
        Eigen::Map<const vec3f>(v[2]) - Eigen::Map<const vec3f>(v[0]);
    area += 0.5 * w1.cross(w2).norm();
  }

  return area;
}
}  // namespace

// Some polygons have zero area for some reason.  When we navigate into a zero
// area polygon, things crash.  So we find all zero area polygons and mark
// them as disabled/not navigable.
void PathFinder::Impl::removeZeroAreaPolys() {
  // Iterate over all tiles
  for (int iTile = 0; iTile < navMesh_->getMaxTiles(); ++iTile) {
    const dtMeshTile* tile =
        const_cast<const dtNavMesh*>(navMesh_.get())->getTile(iTile);
    if (!tile)
      continue;

    // Iterate over all polygons in a tile
    for (int jPoly = 0; jPoly < tile->header->polyCount; ++jPoly) {
      // Get the polygon reference from the tile and polygon id
      dtPolyRef polyRef = navMesh_->encodePolyId(iTile, tile->salt, jPoly);
      const dtPoly* poly = nullptr;
      const dtMeshTile* tmp = nullptr;
      navMesh_->getTileAndPolyByRefUnsafe(polyRef, &tmp, &poly);

      if (polyArea(poly, tile) < 1e-5) {
        navMesh_->setPolyFlags(polyRef, POLYFLAGS_DISABLED);
      }
    }
  }
}

bool PathFinder::Impl::loadNavMesh(const std::string& path) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp)
    return false;

  // Read header.
  NavMeshSetHeader header;
  size_t readLen = fread(&header, sizeof(NavMeshSetHeader), 1, fp);
  if (readLen != 1) {
    fclose(fp);
    return false;
  }
  if (header.magic != NAVMESHSET_MAGIC) {
    fclose(fp);
    return false;
  }
  if (header.version != NAVMESHSET_VERSION) {
    fclose(fp);
    return false;
  }

  vec3f bmin, bmax;

  dtNavMesh* mesh = dtAllocNavMesh();
  if (!mesh) {
    fclose(fp);
    return false;
  }
  dtStatus status = mesh->init(&header.params);
  if (dtStatusFailed(status)) {
    fclose(fp);
    return false;
  }

  // Read tiles.
  for (int i = 0; i < header.numTiles; ++i) {
    NavMeshTileHeader tileHeader;
    readLen = fread(&tileHeader, sizeof(tileHeader), 1, fp);
    if (readLen != 1) {
      fclose(fp);
      return false;
    }

    if (!tileHeader.tileRef || !tileHeader.dataSize)
      break;

    unsigned char* data = static_cast<unsigned char*>(
        dtAlloc(tileHeader.dataSize, DT_ALLOC_PERM));
    if (!data)
      break;
    memset(data, 0, tileHeader.dataSize);
    readLen = fread(data, tileHeader.dataSize, 1, fp);
    if (readLen != 1) {
      dtFree(data);
      fclose(fp);
      return false;
    }

    mesh->addTile(data, tileHeader.dataSize, DT_TILE_FREE_DATA,
                  tileHeader.tileRef, 0);
    const dtMeshTile* tile = mesh->getTileByRef(tileHeader.tileRef);
    if (i == 0) {
      bmin = vec3f(tile->header->bmin);
      bmax = vec3f(tile->header->bmax);
    } else {
      bmin = bmin.array().min(Eigen::Array3f{tile->header->bmin});
      bmax = bmax.array().max(Eigen::Array3f{tile->header->bmax});
    }
  }

  fclose(fp);

  navMesh_.reset(mesh);
  bounds_ = std::make_pair(bmin, bmax);

  removeZeroAreaPolys();

  return initNavQuery();
}

bool PathFinder::Impl::saveNavMesh(const std::string& path) {
  const dtNavMesh* navMesh = navMesh_.get();
  if (!navMesh)
    return false;

  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp)
    return false;

  // Store header.
  NavMeshSetHeader header;
  header.magic = NAVMESHSET_MAGIC;
  header.version = NAVMESHSET_VERSION;
  header.numTiles = 0;
  for (int i = 0; i < navMesh->getMaxTiles(); ++i) {
    const dtMeshTile* tile = navMesh->getTile(i);
    if (!tile || !tile->header || !tile->dataSize)
      continue;
    header.numTiles++;
  }
  memcpy(&header.params, navMesh->getParams(), sizeof(dtNavMeshParams));
  fwrite(&header, sizeof(NavMeshSetHeader), 1, fp);

  // Store tiles.
  for (int i = 0; i < navMesh->getMaxTiles(); ++i) {
    const dtMeshTile* tile = navMesh->getTile(i);
    if (!tile || !tile->header || !tile->dataSize)
      continue;

    NavMeshTileHeader tileHeader;
    tileHeader.tileRef = navMesh->getTileRef(tile);
    tileHeader.dataSize = tile->dataSize;
    fwrite(&tileHeader, sizeof(tileHeader), 1, fp);

    fwrite(tile->data, tile->dataSize, 1, fp);
  }

  fclose(fp);

  return true;
}

void PathFinder::Impl::seed(uint32_t newSeed) {
  // TODO: this should be using core::Random instead, but passing function
  // to navQuery_->findRandomPoint needs to be figured out first
  srand(newSeed);
}

// Returns a random number [0..1]
static float frand() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

vec3f PathFinder::Impl::getRandomNavigablePoint() {
  dtPolyRef ref;
  constexpr float inf = std::numeric_limits<float>::infinity();
  vec3f pt(inf, inf, inf);

  navQuery_->findRandomPoint(filter_.get(), frand, &ref, pt.data());
  return pt;
}

namespace {
float pathLength(const std::vector<vec3f>& points) {
  float length = 0;
  const vec3f* previousPoint = &points[0];
  for (const auto& pt : points) {
    length += (*previousPoint - pt).norm();
    previousPoint = &pt;
  }

  return length;
}
}  // namespace

std::tuple<float, std::vector<vec3f>> PathFinder::Impl::findPathInternal(
    const NavMeshPoint& start,
    const NavMeshPoint& end) {
  // check if trivial path (start is same as end) and early return
  if (start.xyz.isApprox(end.xyz)) {
    return std::make_tuple(0.0f, std::vector<vec3f>{});
  }

  // Check if there is a path between the start and any of the ends
  if (!islandSystem_->hasConnection(start.polyId, end.polyId)) {
    return std::make_tuple(std::numeric_limits<float>::infinity(),
                           std::vector<vec3f>{});
  }

  static const int MAX_POLYS = 256;
  dtPolyRef polys[MAX_POLYS];

  int numPolys = 0;
  dtStatus status = navQuery_->findPath(
      start.polyId, end.polyId, start.xyz.data(), end.xyz.data(), filter_.get(),
      polys, &numPolys, MAX_POLYS);
  if (status != DT_SUCCESS || numPolys == 0) {
    return std::make_tuple(std::numeric_limits<float>::infinity(),
                           std::vector<vec3f>{});
  }

  int numPoints = 0;
  std::vector<vec3f> points(MAX_POLYS);
  status = navQuery_->findStraightPath(start.xyz.data(), end.xyz.data(), polys,
                                       numPolys, points[0].data(), 0, 0,
                                       &numPoints, MAX_POLYS);
  if (status != DT_SUCCESS || numPoints == 0) {
    return std::make_tuple(std::numeric_limits<float>::infinity(),
                           std::vector<vec3f>{});
  }

  points.resize(numPoints);

  const float length = pathLength(points);

  return std::make_tuple(length, std::move(points));
}

bool PathFinder::Impl::findPath(ShortestPath& path) {
  path.geodesicDistance = std::numeric_limits<float>::infinity();
  path.points.clear();

  const std::tuple<float, std::vector<vec3f>> findResult =
      findPathInternal(path.requestedStart, path.requestedEnd);

  if (std::get<0>(findResult) < path.geodesicDistance) {
    path.geodesicDistance = std::get<0>(findResult);
    path.points = std::move(std::get<1>(findResult));
  }

  return path.geodesicDistance < std::numeric_limits<float>::infinity();
}

NavMeshPoint PathFinder::Impl::tryStep(const NavMeshPoint& start,
                                       const esp::vec3f& endXYZ,
                                       bool allowSliding) {
  static const int MAX_POLYS = 256;
  dtPolyRef polys[MAX_POLYS];

  vec3f endPoint;
  int numPolys;
  navQuery_->moveAlongSurface(start.polyId, start.xyz.data(), endXYZ.data(),
                              filter_.get(), endPoint.data(), polys, &numPolys,
                              MAX_POLYS, allowSliding);
  // If there isn't any possible path between start and end, just return
  // start, that is cleanest
  if (numPolys == 0) {
    return start;
  }

  // According to recast's code
  // (https://github.com/recastnavigation/recastnavigation/blob/master/Detour/Source/DetourNavMeshQuery.cpp#L2006-L2007),
  // the endPoint is not guaranteed to be actually on the surface of the
  // navmesh, it seems to be in 99.9% of cases for us, but there are some
  // extreme edge cases where it won't be, so explicitly get the height of the
  // surface at the endPoint and set its height to that.
  // Note, this will never fail as endPoint is always within in the poly
  // polys[numPolys - 1]
  navQuery_->getPolyHeight(polys[numPolys - 1], endPoint.data(), &endPoint[1]);

  return {endPoint, polys[numPolys - 1]};
}

NavMeshPoint PathFinder::Impl::snapPoint(const esp::vec3f& pt) {
  dtStatus status;
  NavMeshPoint navPt;
  std::tie(status, navPt.polyId, navPt.xyz) =
      projectToPoly(pt, navQuery_.get(), filter_.get());

  if (dtStatusSucceed(status)) {
    return navPt;
  } else {
    return {esp::vec3f{NAN, NAN, NAN}, 0};
  }
}

float PathFinder::Impl::islandRadius(const vec3f& pt) const {
  dtPolyRef ptRef;
  dtStatus status;
  std::tie(status, ptRef, std::ignore) =
      projectToPoly(pt, navQuery_.get(), filter_.get());
  if (status != DT_SUCCESS || ptRef == 0) {
    return 0.0;
  } else {
    return islandSystem_->islandRadius(ptRef);
  }
}

bool PathFinder::Impl::isNavigable(const vec3f& pt,
                                   const float maxYDelta /*= 0.5*/) const {
  dtPolyRef ptRef;
  dtStatus status;
  vec3f polyPt;
  std::tie(status, ptRef, polyPt) =
      projectToPoly(pt, navQuery_.get(), filter_.get());

  if (status != DT_SUCCESS || ptRef == 0)
    return false;

  if (std::abs(polyPt[1] - pt[1]) > maxYDelta ||
      (Eigen::Vector2f(pt[0], pt[2]) - Eigen::Vector2f(polyPt[0], polyPt[2]))
              .norm() > 1e-2)
    return false;

  return true;
}

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>
PathFinder::Impl::getTopDownView(const float pixelsPerMeter,
                                 const float height) {
  std::pair<vec3f, vec3f> mapBounds = bounds();
  vec3f bound1 = mapBounds.first;
  vec3f bound2 = mapBounds.second;

  float xspan = std::abs(bound1[0] - bound2[0]);
  float zspan = std::abs(bound1[2] - bound2[2]);
  int xResolution = xspan / pixelsPerMeter;
  int zResolution = zspan / pixelsPerMeter;
  float startx = fmin(bound1[0], bound2[0]);
  float startz = fmin(bound1[2], bound2[2]);
  MatrixXb topdownMap(zResolution, xResolution);

  float curz = startz;
  float curx = startx;
  for (int h = 0; h < zResolution; h++) {
    for (int w = 0; w < xResolution; w++) {
      vec3f point = vec3f(curx, height, curz);
      topdownMap(h, w) = isNavigable(point, 0.5);
      curx = curx + pixelsPerMeter;
    }
    curz = curz + pixelsPerMeter;
    curx = startx;
  }

  return topdownMap;
}

PathFinder::PathFinder() : pimpl_{spimpl::make_unique_impl<Impl>()} {};

vec3f PathFinder::getRandomNavigablePoint() {
  return pimpl_->getRandomNavigablePoint();
}

bool PathFinder::findPath(ShortestPath& path) {
  return pimpl_->findPath(path);
}

NavMeshPoint PathFinder::tryStep(const NavMeshPoint& start,
                                 const esp::vec3f& end) {
  return pimpl_->tryStep(start, end, /*allowSliding=*/true);
}

NavMeshPoint PathFinder::tryStepNoSliding(const NavMeshPoint& start,
                                          const esp::vec3f& end) {
  return pimpl_->tryStep(start, end, /*allowSliding=*/false);
}

NavMeshPoint PathFinder::snapPoint(const esp::vec3f& pt) {
  return pimpl_->snapPoint(pt);
}

bool PathFinder::loadNavMesh(const std::string& path) {
  return pimpl_->loadNavMesh(path);
}

bool PathFinder::saveNavMesh(const std::string& path) {
  return pimpl_->saveNavMesh(path);
}

bool PathFinder::isLoaded() const {
  return pimpl_->isLoaded();
}

void PathFinder::seed(uint32_t newSeed) {
  return pimpl_->seed(newSeed);
}

float PathFinder::islandRadius(const vec3f& pt) const {
  return pimpl_->islandRadius(pt);
}

bool PathFinder::isNavigable(const vec3f& pt, const float maxYDelta) const {
  return pimpl_->isNavigable(pt);
}

std::pair<vec3f, vec3f> PathFinder::bounds() const {
  return pimpl_->bounds();
}

Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> PathFinder::getTopDownView(
    const float pixelsPerMeter,
    const float height) {
  return pimpl_->getTopDownView(pixelsPerMeter, height);
}

}  // namespace nav
}  // namespace esp
