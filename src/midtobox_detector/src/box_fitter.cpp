// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// Known-size 3D cube geometric fitting from partial-view LiDAR point clusters.
// Adapts Zhang IV 2017 L-shape closeness fitting (SS47816/lidar_obstacle_detector, MIT)
// and known-size partial-view centroid correction (autoware_shape_estimation, Apache-2.0),
// explicitly branching 2-face L-shape vs 1-face front-only to avoid corner-flip bias.

#include "midtobox_detector/box_fitter.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace midtobox
{

namespace
{

inline double huberWeight(double residual, double delta)
{
  const double a = std::abs(residual);
  return (a <= delta) ? 1.0 : (delta / a);
}

inline double huberLoss(double residual, double delta)
{
  const double a = std::abs(residual);
  if (a <= delta) {
    return 0.5 * a * a;
  }
  return delta * (a - 0.5 * delta);
}

// Fast O(N) quantile on a scratch copy (modifies buffer in place).
double quantileInPlace(std::vector<double> & buf, double q)
{
  if (buf.empty()) {
    return 0.0;
  }
  if (buf.size() == 1) {
    return buf[0];
  }
  const double clamped_q = std::clamp(q, 0.0, 1.0);
  const double pos = clamped_q * static_cast<double>(buf.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  std::nth_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(lo), buf.end());
  const double v_lo = buf[lo];
  if (hi == lo) {
    return v_lo;
  }
  // Elements after lo are >= v_lo, so min_element in [lo+1, end) equals element at hi == lo+1.
  const double v_hi = *std::min_element(
    buf.begin() + static_cast<std::ptrdiff_t>(lo + 1), buf.end());
  const double frac = pos - static_cast<double>(lo);
  return v_lo + frac * (v_hi - v_lo);
}

// Evaluate Zhang IV 2017 closeness criterion at angle theta (radians).
double evalCloseness(
  const std::vector<Eigen::Vector2d> & pts,
  double theta,
  double d0,
  std::vector<double> & c1_buf,
  std::vector<double> & c2_buf,
  std::vector<double> & scratch)
{
  const double cs = std::cos(theta);
  const double sn = std::sin(theta);
  const size_t n = pts.size();
  c1_buf.resize(n);
  c2_buf.resize(n);
  for (size_t i = 0; i < n; ++i) {
    c1_buf[i] = pts[i].x() * cs + pts[i].y() * sn;
    c2_buf[i] = -pts[i].x() * sn + pts[i].y() * cs;
  }

  // Use 1% trimmed envelope to stay immune to isolated edge-dragging outliers
  scratch = c1_buf;
  const double c1_min = quantileInPlace(scratch, 0.01);
  scratch = c1_buf;
  const double c1_max = quantileInPlace(scratch, 0.99);

  scratch = c2_buf;
  const double c2_min = quantileInPlace(scratch, 0.01);
  scratch = c2_buf;
  const double c2_max = quantileInPlace(scratch, 0.99);

  double score = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d1 = std::min(std::abs(c1_max - c1_buf[i]), std::abs(c1_buf[i] - c1_min));
    const double d2 = std::min(std::abs(c2_max - c2_buf[i]), std::abs(c2_buf[i] - c2_min));
    const double d = std::max(std::min(d1, d2), d0);
    score += 1.0 / d;
  }
  return score;
}

// Fit a single 2D line n·p = d via deterministic RANSAC + Huber IRLS weighted PCA.
// Normal `n_out` is oriented to satisfy n_out · (sensor_2d - mean) >= 0.
bool fitSingleLineRansacHuber(
  const std::vector<Eigen::Vector2d> & pts,
  const Eigen::Vector2d & sensor_2d,
  const BoxFittingConfig & cfg,
  Eigen::Vector2d & n_out,
  double & d_out)
{
  const size_t n = pts.size();
  if (n < 2) {
    return false;
  }

  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> dist_idx(0, n - 1);

  Eigen::Vector2d best_n(1.0, 0.0);
  double best_d = 0.0;
  int best_inliers = -1;
  double best_cost = std::numeric_limits<double>::infinity();

  for (int iter = 0; iter < cfg.ransac_line_iters; ++iter) {
    const size_t i1 = dist_idx(rng);
    const size_t i2 = dist_idx(rng);
    if (i1 == i2) {
      continue;
    }
    const Eigen::Vector2d diff = pts[i2] - pts[i1];
    const double len = diff.norm();
    if (len < 0.04) {
      continue;
    }
    const Eigen::Vector2d cand_n(-diff.y() / len, diff.x() / len);
    const double cand_d = cand_n.dot(pts[i1]);

    int inliers = 0;
    double cost = 0.0;
    for (const auto & p : pts) {
      const double r = std::abs(cand_n.dot(p) - cand_d);
      if (r <= cfg.ransac_line_thresh) {
        ++inliers;
      }
      cost += std::min(r, cfg.ransac_line_thresh * 1.5);
    }
    if (inliers > best_inliers || (inliers == best_inliers && cost < best_cost)) {
      best_inliers = inliers;
      best_cost = cost;
      best_n = cand_n;
      best_d = cand_d;
    }
  }

  if (best_inliers < 2) {
    return false;
  }

  // Collect candidate points near RANSAC line for Huber IRLS refinement
  std::vector<Eigen::Vector2d> inlier_pts;
  inlier_pts.reserve(n);
  const double refine_gate = std::max(cfg.ransac_line_thresh * 2.5, cfg.face_band);
  for (const auto & p : pts) {
    if (std::abs(best_n.dot(p) - best_d) <= refine_gate) {
      inlier_pts.push_back(p);
    }
  }
  if (inlier_pts.size() < 2) {
    inlier_pts = pts;
  }

  // Huber IRLS via weighted 2x2 PCA
  Eigen::Vector2d cur_n = best_n;
  double cur_d = best_d;
  std::vector<double> weights(inlier_pts.size(), 1.0);

  for (int iter = 0; iter < 12; ++iter) {
    double w_sum = 0.0;
    Eigen::Vector2d mu = Eigen::Vector2d::Zero();
    for (size_t i = 0; i < inlier_pts.size(); ++i) {
      const double r = cur_n.dot(inlier_pts[i]) - cur_d;
      weights[i] = huberWeight(r, cfg.huber_delta);
      w_sum += weights[i];
      mu += weights[i] * inlier_pts[i];
    }
    if (w_sum < 1e-9) {
      break;
    }
    mu /= w_sum;

    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    for (size_t i = 0; i < inlier_pts.size(); ++i) {
      const Eigen::Vector2d diff = inlier_pts[i] - mu;
      cov += weights[i] * (diff * diff.transpose());
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
    Eigen::Vector2d next_n = solver.eigenvectors().col(0).normalized();
    if (next_n.dot(cur_n) < 0.0) {
      next_n = -next_n;
    }
    const double next_d = next_n.dot(mu);
    const double step = (next_n - cur_n).norm() + std::abs(next_d - cur_d);
    cur_n = next_n;
    cur_d = next_d;
    if (step < 1e-6) {
      break;
    }
  }

  // Orient normal to point toward the sensor
  Eigen::Vector2d mean_pt = Eigen::Vector2d::Zero();
  for (const auto & p : inlier_pts) {
    mean_pt += p;
  }
  mean_pt /= static_cast<double>(inlier_pts.size());
  if (cur_n.dot(sensor_2d - mean_pt) < 0.0) {
    cur_n = -cur_n;
    cur_d = -cur_d;
  }

  n_out = cur_n;
  d_out = cur_d;
  return true;
}

// Coupled orthogonal 2-line Huber IRLS refinement for 2-face L-shape corner estimation.
// Optimizes (theta, d1', d2') such that n1(theta) = (cos theta, sin theta),
// n2(theta) = s * (-sin theta, cos theta) with s in {-1, +1} preserving handedness.
void refineTwoFaceOrthogonalHuber(
  const std::vector<Eigen::Vector2d> & pts,
  const Eigen::Vector2d & center_ref,
  double & theta,
  double handedness,
  double & d1_rel,
  double & d2_rel,
  const BoxFittingConfig & cfg)
{
  const double outlier_gate = std::max(cfg.face_band * 2.5, 0.06);

  for (int iter = 0; iter < 15; ++iter) {
    const double cs = std::cos(theta);
    const double sn = std::sin(theta);
    const Eigen::Vector2d n1(cs, sn);
    const Eigen::Vector2d n2(handedness * (-sn), handedness * cs);
    // Exact derivatives: dn1/dtheta = handedness * n2, dn2/dtheta = -handedness * n1
    const Eigen::Vector2d dn1 = handedness * n2;
    const Eigen::Vector2d dn2 = -handedness * n1;

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    int used_f1 = 0;
    int used_f2 = 0;

    for (const auto & p : pts) {
      const Eigen::Vector2d q = p - center_ref;
      const double r1 = q.dot(n1) - d1_rel;
      const double r2 = q.dot(n2) - d2_rel;
      const double ar1 = std::abs(r1);
      const double ar2 = std::abs(r2);
      if (std::min(ar1, ar2) > outlier_gate) {
        continue;
      }

      if (ar1 <= ar2) {
        const double w = huberWeight(r1, cfg.huber_delta);
        const Eigen::Vector3d J(q.dot(dn1), -1.0, 0.0);
        H += w * (J * J.transpose());
        g += w * J * r1;
        ++used_f1;
      } else {
        const double w = huberWeight(r2, cfg.huber_delta);
        const Eigen::Vector3d J(q.dot(dn2), 0.0, -1.0);
        H += w * (J * J.transpose());
        g += w * J * r2;
        ++used_f2;
      }
    }

    if (used_f1 < 3 || used_f2 < 3) {
      break;
    }

    H.diagonal().array() += 1e-5;
    const Eigen::Vector3d step = H.ldlt().solve(-g);
    if (!step.allFinite()) {
      break;
    }

    // Clamp angular step per iteration for stability
    const double dtheta = std::clamp(step(0), -0.15, 0.15);
    theta += dtheta;
    d1_rel += step(1);
    d2_rel += step(2);

    if (std::abs(dtheta) < 1e-5 && std::abs(step(1)) < 1e-5 && std::abs(step(2)) < 1e-5) {
      break;
    }
  }
}

// Compute signed residual and world normal from 2D point p to nearest edge of box (c, theta, L, W).
// Returns false if point is outside the max gating distance.
bool computePointToBoxEdge2D(
  const Eigen::Vector2d & p,
  const Eigen::Vector2d & c,
  double theta,
  double half_l,
  double half_w,
  double & signed_residual,
  Eigen::Vector2d & n_world,
  double & unsigned_dist)
{
  const double cs = std::cos(theta);
  const double sn = std::sin(theta);
  const Eigen::Vector2d e1(cs, sn);
  const Eigen::Vector2d e2(-sn, cs);

  const Eigen::Vector2d d = p - c;
  const double lx = d.dot(e1);
  const double ly = d.dot(e2);

  const bool out_x = std::abs(lx) > half_l;
  const bool out_y = std::abs(ly) > half_w;

  if (!out_x && !out_y) {
    // Inside box: snap to closest of the 4 edges
    const double dx_pos = half_l - lx;
    const double dx_neg = lx + half_l;
    const double dy_pos = half_w - ly;
    const double dy_neg = ly + half_w;

    double best = dx_pos;
    int edge = 0;  // +e1
    if (dx_neg < best) { best = dx_neg; edge = 1; }  // -e1
    if (dy_pos < best) { best = dy_pos; edge = 2; }  // +e2
    if (dy_neg < best) { best = dy_neg; edge = 3; }  // -e2

    unsigned_dist = best;
    signed_residual = -best;
    if (edge == 0) {
      n_world = e1;
    } else if (edge == 1) {
      n_world = -e1;
    } else if (edge == 2) {
      n_world = e2;
    } else {
      n_world = -e2;
    }
    return true;
  }

  if (out_x && !out_y) {
    const double sgn = (lx >= 0.0) ? 1.0 : -1.0;
    unsigned_dist = std::abs(lx) - half_l;
    signed_residual = unsigned_dist;
    n_world = sgn * e1;
    return true;
  }

  if (!out_x && out_y) {
    const double sgn = (ly >= 0.0) ? 1.0 : -1.0;
    unsigned_dist = std::abs(ly) - half_w;
    signed_residual = unsigned_dist;
    n_world = sgn * e2;
    return true;
  }

  // Corner region: outside both X and Y
  const double sx = (lx >= 0.0) ? 1.0 : -1.0;
  const double sy = (ly >= 0.0) ? 1.0 : -1.0;
  const double ex = lx - sx * half_l;
  const double ey = ly - sy * half_w;
  const double dist = std::hypot(ex, ey);
  unsigned_dist = dist;
  signed_residual = dist;
  if (dist < 1e-9) {
    n_world = sx * e1;
  } else {
    n_world = (ex / dist) * e1 + (ey / dist) * e2;
  }
  return true;
}

// Pure 2D SE(2) Huber point-to-edge refinement with damped Gauss-Newton.
// For 1-face fits, applies a soft lateral anchor along `tangent_anchor` to prevent unconstrained sliding.
void refineBoxSE2Huber(
  const std::vector<Eigen::Vector2d> & side_pts,
  double half_l,
  double half_w,
  bool is_single_face,
  const Eigen::Vector2d & tangent_anchor,
  const BoxFittingConfig & cfg,
  Eigen::Vector2d & center_io,
  double & theta_io)
{
  const Eigen::Vector2d c0 = center_io;
  const double theta0 = theta_io;
  constexpr double kGateDist = 0.05;

  auto evalObj = [&](const Eigen::Vector2d & c, double th) {
      double obj = 0.0;
      for (const auto & p : side_pts) {
        double r = 0.0;
        double dist = 0.0;
        Eigen::Vector2d n_w;
        computePointToBoxEdge2D(p, c, th, half_l, half_w, r, n_w, dist);
        if (dist <= kGateDist) {
          obj += huberLoss(r, cfg.huber_delta);
        } else {
          obj += huberLoss(kGateDist, cfg.huber_delta);
        }
      }
      if (is_single_face) {
        const double lat_shift = (c - c0).dot(tangent_anchor);
        obj += 0.5 * 50.0 * lat_shift * lat_shift;
      }
      return obj;
    };

  Eigen::Vector2d cur_c = c0;
  double cur_th = theta0;
  double cur_obj = evalObj(cur_c, cur_th);
  double lambda = 1e-2;

  for (int iter = 0; iter < cfg.se2_max_iters; ++iter) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    int inliers = 0;

    for (const auto & p : side_pts) {
      double r = 0.0;
      double dist = 0.0;
      Eigen::Vector2d n_w;
      computePointToBoxEdge2D(p, cur_c, cur_th, half_l, half_w, r, n_w, dist);
      if (dist > kGateDist) {
        continue;
      }
      const double w = huberWeight(r, cfg.huber_delta);
      const Eigen::Vector2d dn_dth(-n_w.y(), n_w.x());
      const Eigen::Vector3d J(-n_w.x(), -n_w.y(), (p - cur_c).dot(dn_dth));
      H += w * (J * J.transpose());
      g += w * J * r;
      ++inliers;
    }

    if (inliers < 10) {
      break;
    }

    if (is_single_face) {
      const double lat_shift = (cur_c - c0).dot(tangent_anchor);
      const Eigen::Vector3d J_lat(tangent_anchor.x(), tangent_anchor.y(), 0.0);
      H += 50.0 * (J_lat * J_lat.transpose());
      g += 50.0 * J_lat * lat_shift;
    }

    H.diagonal().array() += lambda;
    const Eigen::Vector3d step = H.ldlt().solve(-g);
    if (!step.allFinite()) {
      break;
    }

    const Eigen::Vector2d cand_c = cur_c + Eigen::Vector2d(step(0), step(1));
    const double cand_th = cur_th + step(2);
    const double max_ang = cfg.se2_max_angle_deg * kDegToRad;
    if ((cand_c - c0).norm() > cfg.se2_max_shift || std::abs(cand_th - theta0) > max_ang) {
      lambda *= 10.0;
      continue;
    }

    const double cand_obj = evalObj(cand_c, cand_th);
    if (cand_obj < cur_obj) {
      cur_c = cand_c;
      cur_th = cand_th;
      cur_obj = cand_obj;
      lambda = std::max(1e-4, lambda * 0.5);
      if (step.norm() < 1e-5) {
        break;
      }
    } else {
      lambda *= 5.0;
      if (lambda > 1e4) {
        break;
      }
    }
  }

  center_io = cur_c;
  theta_io = cur_th;
}

// Evaluate 3D distance from a point to the surface of a 3D box centered at `c` with yaw `theta`.
double pointToBoxSurface3D(
  const Eigen::Vector3d & p,
  const Eigen::Vector3d & c,
  double theta,
  double half_l,
  double half_w,
  double half_h)
{
  const double cs = std::cos(theta);
  const double sn = std::sin(theta);
  const double dx = p.x() - c.x();
  const double dy = p.y() - c.y();
  const double lx = dx * cs + dy * sn;
  const double ly = -dx * sn + dy * cs;
  const double lz = p.z() - c.z();

  const double ox = std::max(std::abs(lx) - half_l, 0.0);
  const double oy = std::max(std::abs(ly) - half_w, 0.0);
  const double oz = std::max(std::abs(lz) - half_h, 0.0);
  if (ox > 0.0 || oy > 0.0 || oz > 0.0) {
    return std::sqrt(ox * ox + oy * oy + oz * oz);
  }
  const double ix = half_l - std::abs(lx);
  const double iy = half_w - std::abs(ly);
  const double iz = half_h - std::abs(lz);
  return std::min({ix, iy, iz});
}

}  // namespace

double BoxFitter::computeQuantile(std::vector<double> values, double q)
{
  return quantileInPlace(values, q);
}

Obb2D BoxFitter::computeRobustObb2D(const std::vector<Eigen::Vector2d> & pts)
{
  Obb2D obb;
  if (pts.empty()) {
    return obb;
  }
  if (pts.size() == 1) {
    obb.center = pts[0];
    return obb;
  }

  double best_area = std::numeric_limits<double>::infinity();
  std::vector<double> u_buf(pts.size());
  std::vector<double> v_buf(pts.size());
  std::vector<double> scratch;

  for (int deg = 0; deg < 90; deg += 2) {
    const double th = static_cast<double>(deg) * kDegToRad;
    const double cs = std::cos(th);
    const double sn = std::sin(th);
    for (size_t i = 0; i < pts.size(); ++i) {
      u_buf[i] = pts[i].x() * cs + pts[i].y() * sn;
      v_buf[i] = -pts[i].x() * sn + pts[i].y() * cs;
    }
    scratch = u_buf;
    const double u_lo = quantileInPlace(scratch, 0.01);
    scratch = u_buf;
    const double u_hi = quantileInPlace(scratch, 0.99);
    scratch = v_buf;
    const double v_lo = quantileInPlace(scratch, 0.01);
    scratch = v_buf;
    const double v_hi = quantileInPlace(scratch, 0.99);

    const double su = std::max(0.0, u_hi - u_lo);
    const double sv = std::max(0.0, v_hi - v_lo);
    const double area = su * sv;
    if (area < best_area) {
      best_area = area;
      const double u_mid = 0.5 * (u_lo + u_hi);
      const double v_mid = 0.5 * (v_lo + v_hi);
      obb.center = Eigen::Vector2d(u_mid * cs - v_mid * sn, u_mid * sn + v_mid * cs);
      if (su >= sv) {
        obb.major_span = su;
        obb.minor_span = sv;
        obb.yaw = th;
      } else {
        obb.major_span = sv;
        obb.minor_span = su;
        obb.yaw = th + kHalfPi;
      }
    }
  }
  return obb;
}

double BoxFitter::searchLShapeOrientation(
  const std::vector<Eigen::Vector2d> & pts,
  const BoxFittingConfig & cfg)
{
  if (pts.size() < 3) {
    return 0.0;
  }

  std::vector<double> c1_buf;
  std::vector<double> c2_buf;
  std::vector<double> scratch;
  c1_buf.reserve(pts.size());
  c2_buf.reserve(pts.size());
  scratch.reserve(pts.size());

  const double d0 = std::max(cfg.closeness_d0, 1e-4);
  const double coarse_step = std::max(cfg.coarse_step_deg, 0.2) * kDegToRad;
  const double fine_step = std::max(cfg.fine_step_deg, 0.05) * kDegToRad;

  double best_theta = 0.0;
  double best_score = -1.0;

  // Coarse search over [0, pi/2)
  for (double th = 0.0; th < kHalfPi - 1e-6; th += coarse_step) {
    const double score = evalCloseness(pts, th, d0, c1_buf, c2_buf, scratch);
    if (score > best_score) {
      best_score = score;
      best_theta = th;
    }
  }

  // Fine search around coarse winner
  const double fine_window = 1.5 * coarse_step;
  const double start_th = best_theta - fine_window;
  const double end_th = best_theta + fine_window;
  for (double th = start_th; th <= end_th + 1e-9; th += fine_step) {
    const double score = evalCloseness(pts, th, d0, c1_buf, c2_buf, scratch);
    if (score > best_score) {
      best_score = score;
      best_theta = th;
    }
  }

  // Wrap into [0, pi/2)
  best_theta = std::fmod(best_theta, kHalfPi);
  if (best_theta < 0.0) {
    best_theta += kHalfPi;
  }
  return best_theta;
}

bool BoxFitter::fitBoxInGroundFrame(
  const std::vector<Eigen::Vector3d> & cluster_pts,
  const Eigen::Vector3d & sensor_origin_ground,
  const BoxFittingConfig & cfg,
  const GateConfig & gate_cfg,
  BoxDetection & detection_ground,
  std::string & reject_reason)
{
  reject_reason.clear();
  if (static_cast<int>(cluster_pts.size()) < gate_cfg.min_points) {
    reject_reason = "too_few_points";
    return false;
  }

  // 1. Split candidate upper-band vs side-face points in ground-leveled frame G (z = 0 is ground)
  std::vector<double> z_vals;
  z_vals.reserve(cluster_pts.size());
  for (const auto & p : cluster_pts) {
    z_vals.push_back(p.z());
  }
  std::vector<double> z_copy = z_vals;
  const double z_p97 = quantileInPlace(z_copy, 0.97);

  // 2. Symmetric early height consistency gate against known box height H
  const double min_expected_z =
    std::max(gate_cfg.min_z, cfg.height - gate_cfg.height_consistency_tol);
  const double max_expected_z = cfg.height + gate_cfg.height_consistency_tol;
  if (z_p97 > max_expected_z || z_p97 < min_expected_z) {
    reject_reason = "height_mismatch";
    return false;
  }

  std::vector<Eigen::Vector2d> top_pts_2d;
  std::vector<Eigen::Vector2d> side_pts_2d;
  top_pts_2d.reserve(cluster_pts.size() / 2);
  side_pts_2d.reserve(cluster_pts.size());

  const double top_cut = z_p97 - cfg.top_band_thickness;
  const double side_top_cut = z_p97 - cfg.side_top_margin;

  for (const auto & p : cluster_pts) {
    if (p.z() >= top_cut) {
      top_pts_2d.emplace_back(p.x(), p.y());
    }
    if (p.z() >= cfg.side_min_z && p.z() <= side_top_cut) {
      side_pts_2d.emplace_back(p.x(), p.y());
    }
  }

  // Fallback if side band was too sparse (e.g. near-overhead view)
  if (side_pts_2d.size() < 15) {
    side_pts_2d.clear();
    side_pts_2d.reserve(cluster_pts.size());
    for (const auto & p : cluster_pts) {
      side_pts_2d.emplace_back(p.x(), p.y());
    }
  }

  // 3. Zhang IV 2017 L-shape closeness search on 2D side points
  const double theta0 = searchLShapeOrientation(side_pts_2d, cfg);
  const Eigen::Vector2d e1(std::cos(theta0), std::sin(theta0));
  const Eigen::Vector2d e2(-std::sin(theta0), std::cos(theta0));

  // Robust cluster 2D center for sensor-facing normal orientation
  std::vector<double> sx_vals;
  std::vector<double> sy_vals;
  sx_vals.reserve(side_pts_2d.size());
  sy_vals.reserve(side_pts_2d.size());
  for (const auto & p : side_pts_2d) {
    sx_vals.push_back(p.x());
    sy_vals.push_back(p.y());
  }
  const Eigen::Vector2d med_2d(
    quantileInPlace(sx_vals, 0.50),
    quantileInPlace(sy_vals, 0.50));
  const Eigen::Vector2d sensor_2d(sensor_origin_ground.x(), sensor_origin_ground.y());
  const Eigen::Vector2d to_sensor = sensor_2d - med_2d;
  const double to_sensor_norm = std::max(to_sensor.norm(), 1e-6);

  // Orient orthogonal face normals (n1, n2) to point into the half-space facing the sensor
  const double sgn1 = (to_sensor.dot(e1) >= 0.0) ? 1.0 : -1.0;
  const double sgn2 = (to_sensor.dot(e2) >= 0.0) ? 1.0 : -1.0;
  const Eigen::Vector2d n1_init = sgn1 * e1;
  const Eigen::Vector2d n2_init = sgn2 * e2;

  // Project side points into sensor-facing (u, v) coordinates
  std::vector<double> u_vals(side_pts_2d.size());
  std::vector<double> v_vals(side_pts_2d.size());
  for (size_t i = 0; i < side_pts_2d.size(); ++i) {
    u_vals[i] = side_pts_2d[i].dot(n1_init);
    v_vals[i] = side_pts_2d[i].dot(n2_init);
  }

  std::vector<double> u_tmp = u_vals;
  const double u_min = quantileInPlace(u_tmp, 0.04);
  u_tmp = u_vals;
  const double u_max = quantileInPlace(u_tmp, 0.96);

  std::vector<double> v_tmp = v_vals;
  const double v_min = quantileInPlace(v_tmp, 0.04);
  v_tmp = v_vals;
  const double v_max = quantileInPlace(v_tmp, 0.96);

  const double span_u = u_max - u_min;
  const double span_v = v_max - v_min;

  // Count exclusive points on each visible face away from the corner
  std::vector<double> excl_f1_v;
  std::vector<double> excl_f2_u;
  excl_f1_v.reserve(side_pts_2d.size());
  excl_f2_u.reserve(side_pts_2d.size());
  for (size_t i = 0; i < side_pts_2d.size(); ++i) {
    const double du = std::abs(u_max - u_vals[i]);
    const double dv = std::abs(v_max - v_vals[i]);
    if (du <= cfg.face_band && (v_max - v_vals[i]) > cfg.face_band) {
      excl_f1_v.push_back(v_vals[i]);
    }
    if (dv <= cfg.face_band && (u_max - u_vals[i]) > cfg.face_band) {
      excl_f2_u.push_back(u_vals[i]);
    }
  }

  double excl_f1_span = 0.0;
  if (excl_f1_v.size() >= 2) {
    std::vector<double> tmp = excl_f1_v;
    const double lo = quantileInPlace(tmp, 0.05);
    tmp = excl_f1_v;
    const double hi = quantileInPlace(tmp, 0.95);
    excl_f1_span = hi - lo;
  }
  double excl_f2_span = 0.0;
  if (excl_f2_u.size() >= 2) {
    std::vector<double> tmp = excl_f2_u;
    const double lo = quantileInPlace(tmp, 0.05);
    tmp = excl_f2_u;
    const double hi = quantileInPlace(tmp, 0.95);
    excl_f2_span = hi - lo;
  }

  // Require both faces to genuinely face the sensor (incidence < ~85 deg, cos > 0.085)
  const double cos1 = n1_init.dot(to_sensor) / to_sensor_norm;
  const double cos2 = n2_init.dot(to_sensor) / to_sensor_norm;

  const bool is_two_face =
    (span_u >= cfg.two_face_min_span) &&
    (span_v >= cfg.two_face_min_span) &&
    (static_cast<int>(excl_f1_v.size()) >= cfg.two_face_min_pts) &&
    (static_cast<int>(excl_f2_u.size()) >= cfg.two_face_min_pts) &&
    (excl_f1_span >= cfg.two_face_min_excl_span) &&
    (excl_f2_span >= cfg.two_face_min_excl_span) &&
    (cos1 > 0.085) && (cos2 > 0.085);

  Eigen::Vector2d c_2d = Eigen::Vector2d::Zero();
  double box_yaw = 0.0;
  double box_l = cfg.length;
  double box_w = cfg.width;
  FitMode fit_mode = FitMode::kSingleFace;
  Eigen::Vector2d tangent_anchor(0.0, 1.0);
  Eigen::Vector2d bias_push_vec = Eigen::Vector2d::Zero();

  if (is_two_face) {
    // 4A. Two-Face L-Shape Branch: coupled orthogonal Huber IRLS -> corner + L/2 inward push
    fit_mode = FitMode::kTwoFaceLShape;
    double th_n1 = std::atan2(n1_init.y(), n1_init.x());
    // Handedness s in {-1, +1} such that n2 = s * (-sin(th_n1), cos(th_n1))
    const Eigen::Vector2d rot90_n1(-n1_init.y(), n1_init.x());
    const double handedness = (n2_init.dot(rot90_n1) >= 0.0) ? 1.0 : -1.0;

    // Initialize d1_rel, d2_rel using median of exclusive face points relative to med_2d
    std::vector<double> f1_proj;
    std::vector<double> f2_proj;
    for (size_t i = 0; i < side_pts_2d.size(); ++i) {
      const Eigen::Vector2d q = side_pts_2d[i] - med_2d;
      const double du = std::abs(u_max - u_vals[i]);
      const double dv = std::abs(v_max - v_vals[i]);
      if (du <= cfg.face_band && du < dv) {
        f1_proj.push_back(q.dot(n1_init));
      } else if (dv <= cfg.face_band) {
        f2_proj.push_back(q.dot(n2_init));
      }
    }
    double d1_rel = f1_proj.empty() ? (u_max - med_2d.dot(n1_init))
                                    : quantileInPlace(f1_proj, 0.50);
    double d2_rel = f2_proj.empty() ? (v_max - med_2d.dot(n2_init))
                                    : quantileInPlace(f2_proj, 0.50);

    refineTwoFaceOrthogonalHuber(side_pts_2d, med_2d, th_n1, handedness, d1_rel, d2_rel, cfg);

    const Eigen::Vector2d n1(std::cos(th_n1), std::sin(th_n1));
    const Eigen::Vector2d n2(handedness * (-std::sin(th_n1)), handedness * std::cos(th_n1));

    // Match dimensions (L along n1, W along n2) when non-square
    double dim_n1 = cfg.length;
    double dim_n2 = cfg.width;
    if (std::abs(cfg.length - cfg.width) > 0.02) {
      // Face 1 spans along n2 (width dim_n2); Face 2 spans along n1 (length dim_n1)
      const double err_keep = std::abs(span_v - cfg.width) + std::abs(span_u - cfg.length);
      const double err_swap = std::abs(span_v - cfg.length) + std::abs(span_u - cfg.width);
      if (err_swap < err_keep) {
        dim_n1 = cfg.width;
        dim_n2 = cfg.length;
      }
    }

    const Eigen::Vector2d p_corner = med_2d + d1_rel * n1 + d2_rel * n2;
    c_2d = p_corner - 0.5 * dim_n1 * n1 - 0.5 * dim_n2 * n2;
    bias_push_vec = -(n1 + n2);

    // Box x-axis aligned with length dimension
    if (std::abs(dim_n1 - cfg.length) <= std::abs(dim_n2 - cfg.length)) {
      box_yaw = std::atan2(n1.y(), n1.x());
    } else {
      box_yaw = std::atan2(n2.y(), n2.x());
    }
    box_l = cfg.length;
    box_w = cfg.width;
  } else {
    // 4B. Single-Face Front-Only Branch:
    // RANSAC + Huber line normal push -(L/2 + range_bias) + symmetric p5/p95 lateral midpoint.
    // Eliminates Autoware's single-face corner-flip bias!
    fit_mode = FitMode::kSingleFace;
    Eigen::Vector2d n_front;
    double d_front = 0.0;
    if (!fitSingleLineRansacHuber(side_pts_2d, sensor_2d, cfg, n_front, d_front)) {
      reject_reason = "line_fit_failed";
      return false;
    }

    const Eigen::Vector2d t_front(-n_front.y(), n_front.x());
    tangent_anchor = t_front;

    // Project front-face inliers onto tangent axis for symmetric p5/p95 lateral midpoint
    std::vector<double> tau_vals;
    tau_vals.reserve(side_pts_2d.size());
    for (const auto & p : side_pts_2d) {
      if (std::abs(n_front.dot(p) - d_front) <= cfg.face_band) {
        tau_vals.push_back(p.dot(t_front));
      }
    }
    if (tau_vals.size() < 5) {
      tau_vals.clear();
      for (const auto & p : side_pts_2d) {
        tau_vals.push_back(p.dot(t_front));
      }
    }

    std::vector<double> tau_copy = tau_vals;
    const double tau_lo = quantileInPlace(tau_vals, 0.05);
    const double tau_hi = quantileInPlace(tau_copy, 0.95);
    const double tau_mid = 0.5 * (tau_lo + tau_hi);
    const double tau_span = tau_hi - tau_lo;

    // Choose depth dimension for non-square boxes based on observed front width
    double depth_dim = cfg.length;
    if (std::abs(cfg.length - cfg.width) > 0.02) {
      if (std::abs(tau_span - cfg.length) < std::abs(tau_span - cfg.width)) {
        depth_dim = cfg.width;
      }
    }

    const Eigen::Vector2d p_front = d_front * n_front + tau_mid * t_front;
    c_2d = p_front - 0.5 * depth_dim * n_front;
    bias_push_vec = -n_front;

    if (std::abs(depth_dim - cfg.length) < 1e-6) {
      box_yaw = std::atan2(n_front.y(), n_front.x());
    } else {
      box_yaw = std::atan2(t_front.y(), t_front.x());
    }
    box_l = cfg.length;
    box_w = cfg.width;

    // Optional upper-surface depth cross-validation: verify upper points do not extend far behind front face
    if (static_cast<int>(top_pts_2d.size()) >= cfg.top_min_points) {
      std::vector<double> top_depth;
      top_depth.reserve(top_pts_2d.size());
      for (const auto & tp : top_pts_2d) {
        top_depth.push_back(d_front - tp.dot(n_front));
      }
      const double top_far = quantileInPlace(top_depth, 0.95);
      // If upper surface extends dramatically deeper than expected box depth, reject oversized
      if (top_far > depth_dim + 0.16) {
        reject_reason = "top_depth_oversized";
        return false;
      }
    }
  }

  // 5. Optional pure 2D SE(2) Huber point-to-edge refinement (on measured surface geometry)
  if (cfg.enable_se2_refine) {
    refineBoxSE2Huber(
      side_pts_2d,
      0.5 * box_l,
      0.5 * box_w,
      fit_mode == FitMode::kSingleFace,
      tangent_anchor,
      cfg,
      c_2d,
      box_yaw);
  }

  // Canonicalize yaw (4-fold [-45, +45) for square, 2-fold [-90, +90) for rectangle)
  const bool is_square = std::abs(box_l - box_w) < 0.02;
  const double sym_period = is_square ? kHalfPi : kPi;
  box_yaw = canonicalizeSymmetricYaw(box_yaw, sym_period);

  // 5b. Authoritative top-face detection & cz estimation using strictly interior points
  // inset from all 4 side walls. Eliminates both:
  //   (1) false top-face triggers on upright 2-face L-shape views (zero actual top points), and
  //   (2) downward Z bias from dense upper side-wall points contaminating the top Z band.
  const double half_l = 0.5 * box_l;
  const double half_w = 0.5 * box_w;
  const double half_h = 0.5 * cfg.height;
  const double cz_ground = half_h;
  double cz = cz_ground;
  bool has_top_face = false;
  int top_interior_count = 0;
  {
    const double cs = std::cos(box_yaw);
    const double sn = std::sin(box_yaw);
    const double min_half = std::min(half_l, half_w);
    const double wall_inset = std::min(0.030, 0.25 * min_half);
    const double min_med_clearance = std::min(0.045, 0.35 * min_half);
    const double inset_l = std::max(0.015, half_l - wall_inset);
    const double inset_w = std::max(0.015, half_w - wall_inset);
    const double strict_top_cut = z_p97 - std::max(cfg.top_band_thickness, 0.005);

    std::vector<double> interior_top_z;
    std::vector<double> interior_clearances;
    interior_top_z.reserve(top_pts_2d.size());
    interior_clearances.reserve(top_pts_2d.size());
    for (const auto & p : cluster_pts) {
      if (p.z() < strict_top_cut) {
        continue;
      }
      const double dx = p.x() - c_2d.x();
      const double dy = p.y() - c_2d.y();
      const double lx = dx * cs + dy * sn;
      const double ly = -dx * sn + dy * cs;
      if (std::abs(lx) <= inset_l && std::abs(ly) <= inset_w) {
        interior_top_z.push_back(p.z());
        interior_clearances.push_back(std::min(half_l - std::abs(lx), half_w - std::abs(ly)));
      }
    }

    const int min_interior_pts = std::max(cfg.top_min_points, 1);
    if (static_cast<int>(interior_top_z.size()) >= min_interior_pts) {
      std::vector<double> tz_copy = interior_top_z;
      const double z_top_clean = quantileInPlace(tz_copy, 0.50);
      std::vector<double> abs_dev;
      abs_dev.reserve(interior_top_z.size());
      for (double z : interior_top_z) {
        abs_dev.push_back(std::abs(z - z_top_clean));
      }
      const double z_mad = quantileInPlace(abs_dev, 0.50);
      const double med_clearance = quantileInPlace(interior_clearances, 0.50);

      if (z_mad <= cfg.top_max_z_mad && med_clearance >= min_med_clearance) {
        const double cz_clean = z_top_clean - half_h;
        if (std::abs(cz_clean - cz_ground) > gate_cfg.height_consistency_tol) {
          reject_reason = "height_mismatch";
          return false;
        }
        cz = cz_clean;
        has_top_face = true;
        top_interior_count = static_cast<int>(interior_top_z.size());
      }
    }
  }

  // 6. Compute 3D surface inlier ratio and RMSE against measured surface geometry
  const Eigen::Vector3d center_geom(c_2d.x(), c_2d.y(), cz);

  int inlier_count = 0;
  double sum_sq_inlier = 0.0;
  for (const auto & p : cluster_pts) {
    const double d = pointToBoxSurface3D(p, center_geom, box_yaw, half_l, half_w, half_h);
    if (d <= cfg.inlier_dist_thresh) {
      ++inlier_count;
      sum_sq_inlier += d * d;
    }
  }

  const double inlier_ratio =
    static_cast<double>(inlier_count) / static_cast<double>(cluster_pts.size());
  const double rmse =
    (inlier_count > 0) ? std::sqrt(sum_sq_inlier / static_cast<double>(inlier_count)) : 1.0;

  if (inlier_ratio < gate_cfg.min_inlier_ratio) {
    reject_reason = "low_inlier_ratio";
    return false;
  }
  if (rmse > gate_cfg.max_rmse) {
    reject_reason = "high_rmse";
    return false;
  }

  // Apply material-calibrated range bias offset after geometric surface fit & quality checks.
  // For 2-face L-shape views, a radial range shift translates both orthogonal faces along the
  // sensor-to-box ray, moving the fitted corner along -u_ray (unit vector, NOT -(n1+n2)).
  if (fit_mode == FitMode::kTwoFaceLShape) {
    const Eigen::Vector2d ray_2d = c_2d - sensor_2d;
    const double ray_norm = ray_2d.norm();
    if (ray_norm > 1e-6) {
      bias_push_vec = ray_2d / ray_norm;
    } else if (bias_push_vec.norm() > 1e-6) {
      bias_push_vec.normalize();
    }
  }
  const Eigen::Vector2d c_2d_final = c_2d + cfg.range_bias_offset * bias_push_vec;

  // Populate detection in ground-leveled frame G
  detection_ground.center = Eigen::Vector3d(c_2d_final.x(), c_2d_final.y(), cz);
  detection_ground.yaw = box_yaw;
  detection_ground.orientation =
    Eigen::Quaterniond(Eigen::AngleAxisd(box_yaw, Eigen::Vector3d::UnitZ()));
  detection_ground.size = Eigen::Vector3d(box_l, box_w, cfg.height);
  detection_ground.fit_mode = fit_mode;
  detection_ground.has_top_face = has_top_face;
  detection_ground.point_count = static_cast<int>(cluster_pts.size());
  detection_ground.side_point_count = static_cast<int>(side_pts_2d.size());
  detection_ground.top_point_count = top_interior_count;
  detection_ground.inlier_ratio = inlier_ratio;
  detection_ground.rmse = rmse;

  // Composite confidence score in [0, 1]
  const double pts_factor = std::min(1.0, static_cast<double>(cluster_pts.size()) / 150.0);
  const double rmse_factor = std::exp(-rmse / 0.025);
  const double mode_bonus = (fit_mode == FitMode::kTwoFaceLShape) ? 1.0 : 0.92;
  detection_ground.score = std::clamp(
    mode_bonus * (0.45 * inlier_ratio + 0.35 * rmse_factor + 0.20 * pts_factor), 0.0, 1.0);

  return true;
}

}  // namespace midtobox
