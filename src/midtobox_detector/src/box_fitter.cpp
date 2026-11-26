// Copyright 2026 midtobox contributors
// SPDX-License-Identifier: Apache-2.0
//
// 已知尺寸 3D 方块单视角几何拟合实现。
// 借鉴 Zhang IV 2017 L-shape Closeness 拟合（SS47816/lidar_obstacle_detector, MIT）
// 与已知尺寸部分视角质心修正（autoware_shape_estimation, Apache-2.0），
// 显式区分双立面 L 形与单立面正面分支，彻底消除单正面角点跳变偏置。

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

// 基于临时数组原地计算 O(N) 线性插值分位数（会修改传入的 buf 顺序）
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
  // nth_element 保证 [lo+1, end) 均 >= v_lo，因此其中的最小值即为第 hi=lo+1 大元素
  const double v_hi = *std::min_element(
    buf.begin() + static_cast<std::ptrdiff_t>(lo + 1), buf.end());
  const double frac = pos - static_cast<double>(lo);
  return v_lo + frac * (v_hi - v_lo);
}

// 评估给定角度 theta (rad) 下的 Zhang IV 2017 Closeness 贴合度得分（越大越贴合正交边缘）
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

  // 采用 1% 截断包络（p1/p99）替代裸 min/max，免疫孤立边缘拉丝飞点
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

// 用确定性 RANSAC + Huber IRLS 加权 PCA 拟合单条 2D 直线 n·p = d。
// 输出法向量 n_out 统一指向雷达侧，即满足 n_out · (sensor_2d - mean) >= 0。
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

  // 收集 RANSAC 直线附近的候选点，送入 Huber IRLS 做精细拟合
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

  // 基于加权 2x2 PCA 的 Huber 迭代重加权最小二乘（IRLS）
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

  // 统一将法向量定向到指向雷达一侧
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

// 双立面 L 形耦合正交双线 Huber IRLS 联合精修（同时优化共享角度 theta 与两条边截距 d1', d2'）。
// 强制 n1(theta) = (cos theta, sin theta) 与 n2(theta) = s * (-sin theta, cos theta) 严格正交。
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
    // 精确解析导数：dn1/dtheta = handedness * n2, dn2/dtheta = -handedness * n1
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

    // 限制单步角度增量，保证高斯牛顿收敛稳定
    const double dtheta = std::clamp(step(0), -0.15, 0.15);
    theta += dtheta;
    d1_rel += step(1);
    d2_rel += step(2);

    if (std::abs(dtheta) < 1e-5 && std::abs(step(1)) < 1e-5 && std::abs(step(2)) < 1e-5) {
      break;
    }
  }
}

// 计算 2D 点 p 到已知矩形盒 (c, theta, L, W) 最近边界的有符号残差与世界系外法向
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
    // 点落在矩形内部：吸附到距离最近的 4 条边之一
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

  // 角点外区域：X 与 Y 方向同时超出矩形边界
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

// 纯 2D SE(2) Huber 点到边阻尼高斯牛顿精修（绝不引入 Z 轴翻转）。
// 在单立面模式下，沿正面切向 `tangent_anchor` 施加软锚定约束，防止沿切线方向无约束滑动。
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

// 计算 3D 点 p 到中心在 c、偏航角 theta 的 3D 长方体 6 个外表面的最短欧氏距离
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

  // 阶段 1：在 [0, pi/2) 区间做粗步长全局搜索
  for (double th = 0.0; th < kHalfPi - 1e-6; th += coarse_step) {
    const double score = evalCloseness(pts, th, d0, c1_buf, c2_buf, scratch);
    if (score > best_score) {
      best_score = score;
      best_theta = th;
    }
  }

  // 阶段 2：在粗搜最优角度附近做 0.1° 级精细搜索
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

  // 将结果归约到 [0, pi/2)
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

  // 1. 在地面整平系 G（z = 0 为地面）中分离顶部候选带与侧面立面点
  std::vector<double> z_vals;
  z_vals.reserve(cluster_pts.size());
  for (const auto & p : cluster_pts) {
    z_vals.push_back(p.z());
  }
  std::vector<double> z_copy = z_vals;
  const double z_p97 = quantileInPlace(z_copy, 0.97);

  // 2. 对称早期高度一致性校验：即使无顶面可见，也能拒判过矮/过高的干扰物体
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

  // 若侧面带点过少（例如极近距俯视视角），回退使用整簇点做 2D 拟合
  if (side_pts_2d.size() < 15) {
    side_pts_2d.clear();
    side_pts_2d.reserve(cluster_pts.size());
    for (const auto & p : cluster_pts) {
      side_pts_2d.emplace_back(p.x(), p.y());
    }
  }

  // 3. 对侧面 2D 点云执行 Zhang IV 2017 L-shape Closeness 朝向搜索
  const double theta0 = searchLShapeOrientation(side_pts_2d, cfg);
  const Eigen::Vector2d e1(std::cos(theta0), std::sin(theta0));
  const Eigen::Vector2d e2(-std::sin(theta0), std::cos(theta0));

  // 用中位数求鲁棒 2D 中心，辅助确定指向雷达侧的外法向符号
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

  // 将两条正交轴法向量 (n1, n2) 统一指向朝向雷达的半空间
  const double sgn1 = (to_sensor.dot(e1) >= 0.0) ? 1.0 : -1.0;
  const double sgn2 = (to_sensor.dot(e2) >= 0.0) ? 1.0 : -1.0;
  const Eigen::Vector2d n1_init = sgn1 * e1;
  const Eigen::Vector2d n2_init = sgn2 * e2;

  // 将侧面点投影到朝向雷达的 (u, v) 坐标系下，统计各立面跨度与独有点数量
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

  // 统计远离公共角点、明确归属于面 1 与面 2 的独有点（避免角点噪声误判双面）
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

  // 两立面必须同时处于面向雷达的有效入射角内（入射角 < ~85°，cos > 0.085）
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
    // 4A. 双立面 L 形分支：耦合正交双线 Huber IRLS 求精确角点 -> 沿两内法向各推进 L/2、W/2
    fit_mode = FitMode::kTwoFaceLShape;
    double th_n1 = std::atan2(n1_init.y(), n1_init.x());
    // 记录手性符号 s ∈ {-1, +1}，使 n2 = s * (-sin(th_n1), cos(th_n1))
    const Eigen::Vector2d rot90_n1(-n1_init.y(), n1_init.x());
    const double handedness = (n2_init.dot(rot90_n1) >= 0.0) ? 1.0 : -1.0;

    // 用两面各自独有点的投影中位数初始化相对截距 d1_rel, d2_rel
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

    // 若目标为长方形（L != W），按实测跨度自动匹配哪条法向对应长边 L、哪条对应宽边 W
    double dim_n1 = cfg.length;
    double dim_n2 = cfg.width;
    if (std::abs(cfg.length - cfg.width) > 0.02) {
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

    // 将方块局部 X 轴对齐到长度方向 L
    if (std::abs(dim_n1 - cfg.length) <= std::abs(dim_n2 - cfg.length)) {
      box_yaw = std::atan2(n1.y(), n1.x());
    } else {
      box_yaw = std::atan2(n2.y(), n2.x());
    }
    box_l = cfg.length;
    box_w = cfg.width;
  } else {
    // 4B. 单立面正面分支：
    // RANSAC + Huber 拟合正面外法向，切向取对称 p5/p95 中点，沿 -n_front 推进半边长。
    // 彻底避开 Autoware 单面锚定最近角点带来的左右跳变与横向偏置问题！
    fit_mode = FitMode::kSingleFace;
    Eigen::Vector2d n_front;
    double d_front = 0.0;
    if (!fitSingleLineRansacHuber(side_pts_2d, sensor_2d, cfg, n_front, d_front)) {
      reject_reason = "line_fit_failed";
      return false;
    }

    const Eigen::Vector2d t_front(-n_front.y(), n_front.x());
    tangent_anchor = t_front;

    // 将正面内点投影到切线方向，用 p5/p95 对称中点定位正面中心
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

    // 若为长方形，根据正面观测宽度推断沿视线纵深方向的边长
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

    // 顶部纵深交叉校验：若顶部点云向后延伸远超已知块深，则拒判为前后粘连超大目标
    if (static_cast<int>(top_pts_2d.size()) >= cfg.top_min_points) {
      std::vector<double> top_depth;
      top_depth.reserve(top_pts_2d.size());
      for (const auto & tp : top_pts_2d) {
        top_depth.push_back(d_front - tp.dot(n_front));
      }
      const double top_far = quantileInPlace(top_depth, 0.95);
      if (top_far > depth_dim + 0.16) {
        reject_reason = "top_depth_oversized";
        return false;
      }
    }
  }

  // 5. 可选纯 2D SE(2) Huber 点到边联合精修（在实测表面几何上微调中心与偏航角）
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

  // 对称偏航角归约：正方形归约到 [-45°, +45°)，长方形归约到 [-90°, +90°)
  const bool is_square = std::abs(box_l - box_w) < 0.02;
  const double sym_period = is_square ? kHalfPi : kPi;
  box_yaw = canonicalizeSymmetricYaw(box_yaw, sym_period);

  // 5b. 权威顶面判定与质心高度 cz 解算：
  // 仅取严格内缩于四壁内部的顶部点，彻底解决：
  //   (1) 正装双面 L 形在无任何顶面点时误触发顶面判定的问题；
  //   (2) 侧壁上沿高密度点混入顶部带导致 cz 稳定偏低 1.5~2 cm 的系统偏差。
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

  // 6. 在实测表面几何上评估 3D 箱面内点率与均方根误差 (RMSE)
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

  // 7. 施加针对材质反射率标定的系统测距偏置补偿 range_bias_offset。
  // 双立面 L 形下，径向测距偏置会沿雷达->方块单位射线方向整体平移两正交面（不能用 -(n1+n2) 否则放大 √2 倍）。
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

  // 汇总填充地面整平系 G 下的单帧检测结果
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

  // 计算综合置信度得分 ∈ [0, 1]（双立面观测稳定性略高于单立面）
  const double pts_factor = std::min(1.0, static_cast<double>(cluster_pts.size()) / 150.0);
  const double rmse_factor = std::exp(-rmse / 0.025);
  const double mode_bonus = (fit_mode == FitMode::kTwoFaceLShape) ? 1.0 : 0.92;
  detection_ground.score = std::clamp(
    mode_bonus * (0.45 * inlier_ratio + 0.35 * rmse_factor + 0.20 * pts_factor), 0.0, 1.0);

  return true;
}

}  // namespace midtobox
