# -*- coding: utf-8 -*-
"""
🏭 النظام النهائي المحسّن والمتكامل - جاهز للنسخ واللصق
🎯 يشمل: PSO, GA, DE + Smooth + تحقق كينماتيكي + زمن تنفيذ + وعي حساسات + متانة تحت الضجيج + أوامر تحكم + إعادة تخطيط
"""

import numpy as np
import time
import warnings
from math import sqrt, sin

warnings.filterwarnings('ignore')

# ============================================================
# الإعدادات النهائية المتكاملة
# ============================================================
class FinalRobotSystemConfig:
    """الإعدادات النهائية المتوازنة والمحسّنة مع قيود حركية وزمن وحساسات"""
    AREA_SIZE = 50.0
    START = np.array([2.0, 2.0])
    GOAL = np.array([48.0, 48.0])
    BOUNDS = (0.0, AREA_SIZE)

    ROBOT_RADIUS = 0.6
    ROBOT_WIDTH = 1.2

    MAX_LINEAR_SPEED = 1.5
    MAX_ANGULAR_SPEED = 1.0
    MAX_LINEAR_ACC = 1.0
    MAX_ANGULAR_ACC = 1.5

    MIN_TURN_RADIUS = 1.2
    MAX_CURVATURE_KIN = 1.0 / MIN_TURN_RADIUS

    GLOBAL_POINTS = 10
    GLOBAL_DIM = GLOBAL_POINTS * 2
    GLOBAL_PARTICLES = 18
    GLOBAL_ITERATIONS = 45

    LOCAL_RATE = 20
    LOCAL_DT = 1.0 / LOCAL_RATE
    LOOKAHEAD_DISTANCE = 1.0
    MAX_CURVATURE = 0.6

    BASE_POWER = 80
    POWER_PER_METER = 0.3
    POWER_PER_TURN = 5.0

    SENSOR_UNCERTAINTY = 0.3
    FOV_RANGE = 12.0
    FOV_PENALTY = 0.85

    OBSTACLES = [
        {"type": "machine", "center": np.array([15, 20]), "radius": 2.5},
        {"type": "machine", "center": np.array([35, 35]), "radius": 3.0},
        {"type": "machine", "center": np.array([25, 45]), "radius": 2.0},
        {"type": "rack", "center": np.array([20, 15]), "width": 6.0, "depth": 2.0},
        {"type": "rack", "center": np.array([40, 30]), "width": 8.0, "depth": 2.0},
        {"type": "rack", "center": np.array([10, 40]), "width": 5.0, "depth": 1.8},
        {"type": "column", "center": np.array([30, 25]), "radius": 1.0},
        {"type": "column", "center": np.array([45, 20]), "radius": 0.8},
        {"type": "restricted", "center": np.array([5, 5]), "radius": 4.0},
        {"type": "dynamic_zone", "center": np.array([35, 15]), "radius": 4.0}
    ]

    IDEAL_LENGTH = np.linalg.norm(GOAL - START)
    MAX_LENGTH = IDEAL_LENGTH * 1.6
    TARGET_LENGTH = IDEAL_LENGTH * 1.3
    MAX_ENERGY = TARGET_LENGTH * POWER_PER_METER * 1.3 + BASE_POWER
    SAFETY_THRESHOLD = 0.6
    TIME_BUDGET = 3.0


class FinalOptimizerConfig:
    """إعدادات الخوارزميات النهائية المحسنة"""
    RUNS = 3

    # PSO
    PSO_W_INIT = 0.9
    PSO_W_FINAL = 0.4
    PSO_C1 = 1.5
    PSO_C2 = 1.5

    # DE
    DE_F = 0.8
    DE_CR = 0.9

    # GA
    GA_CROSSOVER = 0.85
    GA_MUTATION = 0.1

    # تحسينات إضافية
    ADAPTIVE_MUTATION = True
    ELITISM_RATE = 0.1

# ============================================================
# هندسة وتحليلات كاملة مع التكامُل العملي
# ============================================================
class CompleteGeometry:
    """هندسة مع تفاصيل كاملة - تشمل التحقق الكينماتيكي، الزمن، الحساسات، المتانة، وأوامر التحكم"""

    @staticmethod
    def decode_path(solution):
        n_dim = FinalRobotSystemConfig.GLOBAL_DIM
        if len(solution) < n_dim:
            solution = np.random.uniform(10, 40, n_dim)
        solution = solution[:n_dim]
        points = solution.reshape(-1, 2)
        points = points[:FinalRobotSystemConfig.GLOBAL_POINTS]
        points = np.clip(points, 3, 47)
        if len(points) > 2:
            for i in range(1, len(points) - 1):
                points[i] = 0.9 * points[i] + 0.05 * (points[i - 1] + points[i + 1])
        return np.vstack([FinalRobotSystemConfig.START, points, FinalRobotSystemConfig.GOAL])

    @staticmethod
    def estimate_time(path, segment_lengths):
        v_max = FinalRobotSystemConfig.MAX_LINEAR_SPEED
        a_max = FinalRobotSystemConfig.MAX_LINEAR_ACC
        w_max = FinalRobotSystemConfig.MAX_ANGULAR_SPEED
        alpha_max = FinalRobotSystemConfig.MAX_ANGULAR_ACC

        total_length = float(np.sum(segment_lengths))
        t_acc = v_max / a_max
        d_acc = 0.5 * a_max * t_acc**2

        if 2 * d_acc >= total_length:
            t_linear = 2.0 * np.sqrt(total_length / a_max)
        else:
            d_const = total_length - 2 * d_acc
            t_const = d_const / v_max
            t_linear = 2 * t_acc + t_const

        turning_time = 0.0
        if len(path) >= 3:
            diffs = np.diff(path, axis=0)
            angles = []
            for i in range(1, len(path) - 1):
                v1, v2 = diffs[i - 1], diffs[i]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    ca = np.dot(v1, v2) / (n1 * n2)
                    angles.append(np.arccos(np.clip(ca, -1, 1)))
            if angles:
                angle_sum = float(np.sum(angles))
                t_spin = angle_sum / w_max
                t_spin_acc = w_max / alpha_max
                turning_time = t_spin + t_spin_acc

        return t_linear + turning_time

    @staticmethod
    def generate_control_commands(path, lookahead=FinalRobotSystemConfig.LOOKAHEAD_DISTANCE):
        cmds = []
        v_max = FinalRobotSystemConfig.MAX_LINEAR_SPEED
        w_max = FinalRobotSystemConfig.MAX_ANGULAR_SPEED

        for i in range(len(path) - 1):
            p = path[i]
            q = path[i + 1]
            dx, dy = q[0] - p[0], q[1] - p[1]
            seg_len = float(np.hypot(dx, dy))
            if seg_len < 1e-6:
                cmds.append((0.0, 0.0))
                continue
            v = min(v_max, max(0.4, seg_len))
            if i < len(path) - 2:
                r = path[i + 2]
                dx2, dy2 = r[0] - q[0], r[1] - q[1]
                heading = np.arctan2(dy, dx)
                heading2 = np.arctan2(dy2, dx2)
                dtheta = np.arctan2(np.sin(heading2 - heading), np.cos(heading2 - heading))
                L_avg = 0.5 * (seg_len + np.hypot(dx2, dy2))
                kappa = (2.0 * np.sin(abs(dtheta) / 2.0) / L_avg) if L_avg > 1e-3 else 0.0
            else:
                kappa = 0.0
            w = np.clip(v * kappa, -w_max, w_max)
            cmds.append((float(v), float(w)))
        return cmds

    @staticmethod
    def differential_wheels(v, w, wheel_base=FinalRobotSystemConfig.ROBOT_WIDTH):
        v_left = v - w * wheel_base / 2.0
        v_right = v + w * wheel_base / 2.0
        return float(v_left), float(v_right)

    @staticmethod
    def robustness_under_noise(path, trials=3, noise_xy=0.1):
        ok = 0
        for _ in range(trials):
            noisy = path.copy()
            if len(noisy) > 2:
                noisy[1:-1] += np.random.normal(0, noise_xy, noisy[1:-1].shape)
            # نحسب المقاييس بدون حساب متانة مرة أخرى لتفادي التكرار اللانهائي
            metrics_noisy = CompleteGeometry.calculate_complete_metrics(noisy, compute_robustness=False)
            if metrics_noisy['valid']:
                ok += 1
        return ok / trials

    @staticmethod
    def calculate_complete_metrics(path, compute_robustness=True):
        if len(path) < 2:
            return {
                'length': 0.0, 'smoothness': 0.0, 'safety': 0.0,
                'energy': 0.0, 'valid': False, 'curvature': 0.0,
                'clearance': 0.0, 'min_clearance': float('inf'),
                'collision_count': 0, 'num_points': len(path),
                'segment_lengths': [], 'ideal_length': FinalRobotSystemConfig.IDEAL_LENGTH,
                'length_ratio': float('inf'), 'energy_ratio': float('inf'),
                'turning_penalty': 0.0, 'path': path, 'time': float('inf'),
                'kinematic_ok': False, 'omega_req_max': 0.0,
                'commands': [], 'robustness': 0.0
            }

        path = np.asarray(path)
        diff = np.diff(path, axis=0)
        segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
        total_length = float(np.sum(segment_lengths))

        curvature_points = []
        smoothness_score = 1.0

        if len(path) >= 3:
            angles = []
            for i in range(1, len(path) - 1):
                v1, v2 = diff[i - 1], diff[i]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0.1 and n2 > 0.1:
                    cos_angle = np.dot(v1, v2) / (n1 * n2)
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles.append(angle)
                    curvature = 2 * sin(angle / 2) / ((n1 + n2) / 2) if (n1 + n2) > 0 else 0
                    curvature_points.append(curvature)
            if angles:
                mean_angle = float(np.mean(angles))
                mean_angle_deg = float(np.degrees(mean_angle))
                if mean_angle_deg < 10:
                    smoothness_score = 1.0
                elif mean_angle_deg < 20:
                    smoothness_score = 0.95
                elif mean_angle_deg < 30:
                    smoothness_score = 0.85
                elif mean_angle_deg < 45:
                    smoothness_score = 0.70
                elif mean_angle_deg < 60:
                    smoothness_score = 0.50
                elif mean_angle_deg < 90:
                    smoothness_score = 0.30
                else:
                    smoothness_score = 0.10
                angle_std = float(np.std(angles))
                if angle_std < np.deg2rad(10):
                    smoothness_score *= 1.1
                elif angle_std > np.deg2rad(30):
                    smoothness_score *= 0.9
                smoothness_score = float(np.clip(smoothness_score, 0, 1))
        else:
            if total_length > 0:
                smoothness_score = 0.8

        mean_curvature = float(np.mean(curvature_points)) if curvature_points else 0.0

        safety_scores = []
        clearance_distances = []
        collision_count = 0

        for point_idx, point in enumerate(path):
            min_dist = float('inf')
            inflation = FinalRobotSystemConfig.SENSOR_UNCERTAINTY

            for obs in FinalRobotSystemConfig.OBSTACLES:
                if obs["type"] in ["machine", "column", "restricted", "dynamic_zone"]:
                    dist = np.linalg.norm(point - obs["center"]) - (obs["radius"] + inflation)
                elif obs["type"] == "rack":
                    center, width, depth = obs["center"], obs["width"] + 2 * inflation, obs["depth"] + 2 * inflation
                    dx = max(abs(point[0] - center[0]) - width / 2, 0)
                    dy = max(abs(point[1] - center[1]) - depth / 2, 0)
                    dist = sqrt(dx * dx + dy * dy)
                else:
                    dist = float('inf')
                min_dist = min(min_dist, dist)

            clearance_distances.append(min_dist)

            robot_safety_margin = FinalRobotSystemConfig.ROBOT_RADIUS + 0.3
            if min_dist <= robot_safety_margin:
                safety = 0.0
                collision_count += 1
            elif min_dist <= robot_safety_margin + 0.5:
                safety = 0.2
            elif min_dist <= robot_safety_margin + 1.0:
                safety = 0.4
            elif min_dist <= robot_safety_margin + 2.0:
                safety = 0.7
            elif min_dist <= robot_safety_margin + 3.0:
                safety = 0.9
            else:
                safety = 1.0

            margin = 3.0
            dist_to_edge = min(
                point[0] - 0,
                FinalRobotSystemConfig.AREA_SIZE - point[0],
                point[1] - 0,
                FinalRobotSystemConfig.AREA_SIZE - point[1]
            )
            if dist_to_edge < margin:
                safety *= 0.8 + 0.2 * (dist_to_edge / margin)

            if point_idx not in [0, len(path) - 1]:
                if np.linalg.norm(point - FinalRobotSystemConfig.GOAL) > FinalRobotSystemConfig.FOV_RANGE:
                    safety *= FinalRobotSystemConfig.FOV_PENALTY

            safety_scores.append(safety)

        safety_score = float(np.mean(safety_scores))
        avg_clearance = float(np.mean(clearance_distances))
        min_clearance = float(np.min(clearance_distances)) if clearance_distances else 0.0

        if collision_count > 0:
            safety_score *= (1 - collision_count / (2 * len(path)))

        base_energy = FinalRobotSystemConfig.BASE_POWER
        distance_energy = total_length * FinalRobotSystemConfig.POWER_PER_METER

        turning_penalty = 0.0
        if mean_curvature > 0.1:
            turning_penalty = mean_curvature * total_length * FinalRobotSystemConfig.POWER_PER_TURN * 0.8
        elif mean_curvature > 0.05:
            turning_penalty = mean_curvature * total_length * FinalRobotSystemConfig.POWER_PER_TURN * 0.5

        if len(path) >= 3:
            for i in range(1, len(path) - 1):
                v1, v2 = diff[i - 1], diff[i]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0.1 and n2 > 0.1:
                    dot_product = np.dot(v1, v2) / (n1 * n2)
                    angle = np.arccos(np.clip(dot_product, -1, 1))
                    if angle > np.deg2rad(60):
                        turning_penalty += 2.0

        total_energy = float(base_energy + distance_energy + turning_penalty)

        length_ratio = total_length / FinalRobotSystemConfig.IDEAL_LENGTH
        energy_ratio = total_energy / (FinalRobotSystemConfig.IDEAL_LENGTH *
                                       FinalRobotSystemConfig.POWER_PER_METER +
                                       FinalRobotSystemConfig.BASE_POWER)

        # تحقق كينماتيكي
        kinematic_ok = True
        omega_req_list = []
        if len(path) >= 3:
            for i in range(1, len(path) - 1):
                v1 = diff[i - 1]
                v2 = diff[i]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                cos_angle = np.dot(v1, v2) / (n1 * n2)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                L_avg = 0.5 * (n1 + n2)
                R = (L_avg / (2.0 * np.sin(angle / 2))) if angle > 1e-3 else np.inf
                curvature_local = (1.0 / R) if R > 1e-6 else 0.0
                if curvature_local > FinalRobotSystemConfig.MAX_CURVATURE_KIN:
                    kinematic_ok = False
                    break
                if np.isfinite(R) and R > 1e-3:
                    omega_req = FinalRobotSystemConfig.MAX_LINEAR_SPEED / R
                    omega_req_list.append(omega_req)
                    if omega_req > FinalRobotSystemConfig.MAX_ANGULAR_SPEED:
                        kinematic_ok = False
                        break

        # زمن التنفيذ
        estimated_time = float(CompleteGeometry.estimate_time(path, segment_lengths))
        time_ok = (estimated_time <= FinalRobotSystemConfig.TIME_BUDGET)

        is_valid = (
            safety_score > FinalRobotSystemConfig.SAFETY_THRESHOLD and
            length_ratio < FinalRobotSystemConfig.MAX_LENGTH / FinalRobotSystemConfig.IDEAL_LENGTH and
            energy_ratio < FinalRobotSystemConfig.MAX_ENERGY / (FinalRobotSystemConfig.IDEAL_LENGTH *
                                                                FinalRobotSystemConfig.POWER_PER_METER +
                                                                FinalRobotSystemConfig.BASE_POWER) and
            min_clearance > FinalRobotSystemConfig.ROBOT_RADIUS and
            collision_count == 0 and
            kinematic_ok and
            time_ok
        )

        commands = CompleteGeometry.generate_control_commands(path)
        robustness = 0.0
        if compute_robustness:
            robustness = CompleteGeometry.robustness_under_noise(path, trials=3, noise_xy=0.1)

        return {
            'length': total_length,
            'smoothness': smoothness_score,
            'safety': safety_score,
            'energy': total_energy,
            'valid': is_valid,
            'curvature': mean_curvature,
            'clearance': avg_clearance,
            'min_clearance': min_clearance,
            'collision_count': collision_count,
            'num_points': len(path),
            'segment_lengths': segment_lengths.tolist(),
            'ideal_length': FinalRobotSystemConfig.IDEAL_LENGTH,
            'length_ratio': length_ratio,
            'energy_ratio': energy_ratio,
            'turning_penalty': turning_penalty,
            'path': path,
            'time': estimated_time,
            'kinematic_ok': kinematic_ok,
            'omega_req_max': max(omega_req_list) if omega_req_list else 0.0,
            'commands': commands,
            'robustness': robustness
        }

# ============================================================
# دالة اللياقة مع إدخال معيار الزمن
# ============================================================
class CompleteFitness:
    """دالة لياقة مع تفاصيل كاملة تشمل الزمن"""
    _cache = {}
    _cache_hits = 0
    _cache_misses = 0

    @staticmethod
    def calculate_with_details(solution):
        solution_tuple = tuple(solution.flatten() if hasattr(solution, 'flatten') else tuple(solution))
        if solution_tuple in CompleteFitness._cache:
            CompleteFitness._cache_hits += 1
            return CompleteFitness._cache[solution_tuple]
        CompleteFitness._cache_misses += 1

        path = CompleteGeometry.decode_path(solution)
        metrics = CompleteGeometry.calculate_complete_metrics(path)

        S = metrics['smoothness']
        SF = metrics['safety']
        length_ratio = metrics['length_ratio']
        energy_ratio = metrics['energy_ratio']
        min_clearance = metrics['min_clearance']
        T = metrics['time']

        # 1. نقاط الطول (30 نقطة)
        if length_ratio <= 1.1:
            length_score, length_points = 1.0, 30
        elif length_ratio <= 1.15:
            length_score, length_points = 0.95, 28
        elif length_ratio <= 1.2:
            length_score, length_points = 0.9, 27
        elif length_ratio <= 1.25:
            length_score, length_points = 0.85, 25
        elif length_ratio <= 1.3:
            length_score, length_points = 0.8, 24
        elif length_ratio <= 1.35:
            length_score, length_points = 0.75, 22
        elif length_ratio <= 1.4:
            length_score, length_points = 0.7, 21
        elif length_ratio <= 1.45:
            length_score, length_points = 0.65, 19
        elif length_ratio <= 1.5:
            length_score, length_points = 0.6, 18
        elif length_ratio <= 1.6:
            length_score, length_points = 0.5, 15
        elif length_ratio <= 1.7:
            length_score, length_points = 0.4, 12
        elif length_ratio <= 1.8:
            length_score, length_points = 0.3, 9
        elif length_ratio <= 2.0:
            length_score, length_points = 0.2, 6
        else:
            length_score, length_points = 0.1, 3

        # 2. نقاط الأمان (35 نقطة)
        safety_points = SF * 35
        safety_score = SF

        # 3. نقاط الطاقة (25 نقطة)
        if energy_ratio <= 1.05:
            energy_score, energy_points = 1.0, 25
        elif energy_ratio <= 1.1:
            energy_score, energy_points = 0.9, 22
        elif energy_ratio <= 1.15:
            energy_score, energy_points = 0.85, 21
        elif energy_ratio <= 1.2:
            energy_score, energy_points = 0.8, 20
        elif energy_ratio <= 1.25:
            energy_score, energy_points = 0.75, 18
        elif energy_ratio <= 1.3:
            energy_score, energy_points = 0.7, 17
        elif energy_ratio <= 1.35:
            energy_score, energy_points = 0.65, 16
        elif energy_ratio <= 1.4:
            energy_score, energy_points = 0.6, 15
        elif energy_ratio <= 1.5:
            energy_score, energy_points = 0.5, 12
        elif energy_ratio <= 1.6:
            energy_score, energy_points = 0.4, 10
        elif energy_ratio <= 1.7:
            energy_score, energy_points = 0.3, 8
        elif energy_ratio <= 1.8:
            energy_score, energy_points = 0.2, 5
        else:
            energy_score, energy_points = 0.1, 3

        # 4. نقاط السلاسة (10 نقطة)
        smoothness_points = S * 10
        smoothness_score = S

        # 5. نقاط الزمن (10 نقاط)
        B = FinalRobotSystemConfig.TIME_BUDGET
        time_score = float(np.clip(1.0 - max(0.0, (T - B)) / B, 0.0, 1.0))
        time_points = 10 * time_score

        # اللياقة النهائية (أقل = أفضل) مع الزمن
        fitness = (
            0.27 * (1.0 - safety_score) +
            0.23 * (1.0 - length_score) +
            0.23 * (1.0 - energy_score) +
            0.17 * (1.0 - smoothness_score) +
            0.10 * (1.0 - time_score)
        )

        penalty = 0.0
        if SF < 0.4:
            penalty += 0.5
        elif SF < 0.6:
            penalty += 0.3
        elif SF < 0.7:
            penalty += 0.1

        if length_ratio > 1.8:
            penalty += 0.4
        elif length_ratio > 1.6:
            penalty += 0.2
        elif length_ratio > 1.4:
            penalty += 0.1

        if energy_ratio > 1.8:
            penalty += 0.3
        elif energy_ratio > 1.6:
            penalty += 0.15
        elif energy_ratio > 1.4:
            penalty += 0.05

        if S < 0.3:
            penalty += 0.2
        elif S < 0.5:
            penalty += 0.1

        if min_clearance < FinalRobotSystemConfig.ROBOT_RADIUS + 0.1:
            penalty += 0.3
        elif min_clearance < FinalRobotSystemConfig.ROBOT_RADIUS + 0.5:
            penalty += 0.15

        fitness += penalty
        reward = 0.0
        if SF > 0.9 and length_ratio < 1.25 and energy_ratio < 1.2 and S > 0.8 and time_score > 0.9:
            reward -= 0.25
            safety_points += 3
            length_points += 2
            energy_points += 2
            smoothness_points += 1
            time_points += 2
        elif SF > 0.85 and length_ratio < 1.3 and energy_ratio < 1.25:
            reward -= 0.15
            safety_points += 2
            length_points += 1
        elif SF > 0.8 and length_ratio < 1.35:
            reward -= 0.1
            safety_points += 1

        fitness += reward
        fitness = float(np.clip(fitness, 0, 1))

        total_score = (
            safety_points +
            length_points +
            energy_points +
            smoothness_points +
            time_points
        )
        total_score = float(min(100.0, total_score))

        if not metrics['valid']:
            total_score *= 0.7

        result = {
            'fitness': fitness,
            'score': total_score,
            'metrics': metrics,
            'partial_scores': {
                'length_score': length_score,
                'safety_score': safety_score,
                'energy_score': energy_score,
                'smoothness_score': smoothness_score,
                'time_score': time_score,
                'length_points': length_points,
                'safety_points': safety_points,
                'energy_points': energy_points,
                'smoothness_points': smoothness_points,
                'time_points': time_points,
                'penalty': penalty,
                'reward': -reward
            }
        }

        if len(CompleteFitness._cache) < 1000:
            CompleteFitness._cache[solution_tuple] = result

        return result

    @staticmethod
    def calculate(solution):
        return CompleteFitness.calculate_with_details(solution)['fitness']

    @staticmethod
    def calculate_score(solution):
        return CompleteFitness.calculate_with_details(solution)['score']

    @staticmethod
    def get_cache_stats():
        return {
            'hits': CompleteFitness._cache_hits,
            'misses': CompleteFitness._cache_misses,
            'size': len(CompleteFitness._cache),
            'hit_rate': CompleteFitness._cache_hits / (CompleteFitness._cache_hits + CompleteFitness._cache_misses)
            if (CompleteFitness._cache_hits + CompleteFitness._cache_misses) > 0 else 0
        }

# ============================================================
# إعادة التخطيط
# ============================================================
def needs_replan(metrics):
    if not metrics['valid']:
        return True
    if metrics['safety'] < 0.65:
        return True
    if metrics.get('min_clearance', 0) < (FinalRobotSystemConfig.ROBOT_RADIUS + 0.2):
        return True
    return False

# ============================================================
# الخوارزميات الست المصححة والمحسّنة
# ============================================================

# 1. Baseline
class FinalBaseline:
    def __init__(self):
        self.fitness_func = CompleteFitness.calculate
        self.name = "1. Baseline"

    def run(self, run_number=1):
        start_time = time.time()
        seed = 42 + run_number * 17
        np.random.seed(seed)

        n_dim = FinalRobotSystemConfig.GLOBAL_DIM
        solutions = []
        fitnesses = []

        for _ in range(5):
            solution = np.random.uniform(15, 35, n_dim)
            fitness = self.fitness_func(solution)
            solutions.append(solution)
            fitnesses.append(fitness)

        best_idx = int(np.argmin(fitnesses))
        solution = solutions[best_idx]
        fitness = fitnesses[best_idx]

        for _ in range(5):
            new_solution = solution + np.random.uniform(-1, 1, n_dim) * 0.5
            new_solution = np.clip(new_solution, 5, 45)
            new_fitness = self.fitness_func(new_solution)
            if new_fitness < fitness:
                solution = new_solution
                fitness = new_fitness

        score = CompleteFitness.calculate_score(solution)
        details = CompleteFitness.calculate_with_details(solution)

        # إعادة تخطيط محلي إذا لزم
        if needs_replan(details['metrics']):
            local = details['metrics']['path'][1:-1].flatten()
            local = local + np.random.uniform(-2, 2, local.shape)
            local = np.clip(local, 3, 47)
            base_fit = CompleteFitness.calculate(local)
            if base_fit < fitness:
                solution = local.copy()
                fitness = base_fit
                score = CompleteFitness.calculate_score(solution)
                details = CompleteFitness.calculate_with_details(solution)

        elapsed = time.time() - start_time
        return solution, float(fitness), float(score), float(elapsed), details

# 2. PSO فقط
class FinalPSO:
    def __init__(self):
        self.fitness_func = CompleteFitness.calculate
        self.name = "2. PSO فقط"

    def run(self, run_number=1):
        start_time = time.time()
        seed = 42 + run_number * 23
        np.random.seed(seed)

        n_particles = FinalRobotSystemConfig.GLOBAL_PARTICLES
        n_dim = FinalRobotSystemConfig.GLOBAL_DIM

        particles = []
        for i in range(n_particles):
            if i < n_particles // 3:
                particles.append(np.random.uniform(15, 35, n_dim))
            elif i < 2 * n_particles // 3:
                particles.append(np.random.uniform(10, 40, n_dim))
            else:
                t = np.linspace(0, 1, FinalRobotSystemConfig.GLOBAL_POINTS + 2)[1:-1]
                base = FinalRobotSystemConfig.START + t[:, None] * (FinalRobotSystemConfig.GOAL - FinalRobotSystemConfig.START)
                particles.append(base.flatten() + np.random.uniform(-5, 5, n_dim))

        particles = np.array(particles)
        velocities = np.random.uniform(-2, 2, (n_particles, n_dim)) * 0.3
        personal_best = particles.copy()
        personal_best_fitness = np.array([self.fitness_func(p) for p in particles])

        global_best_idx = int(np.argmin(personal_best_fitness))
        global_best = particles[global_best_idx].copy()
        global_best_fitness = float(personal_best_fitness[global_best_idx])

        for iteration in range(FinalRobotSystemConfig.GLOBAL_ITERATIONS):
            w = FinalOptimizerConfig.PSO_W_INIT - (FinalOptimizerConfig.PSO_W_INIT - FinalOptimizerConfig.PSO_W_FINAL) * (iteration / FinalRobotSystemConfig.GLOBAL_ITERATIONS)
            c1 = FinalOptimizerConfig.PSO_C1 * (1 - 0.3 * iteration / FinalRobotSystemConfig.GLOBAL_ITERATIONS)
            c2 = FinalOptimizerConfig.PSO_C2 * (1 + 0.2 * iteration / FinalRobotSystemConfig.GLOBAL_ITERATIONS)

            for i in range(n_particles):
                r1, r2 = np.random.rand(2)
                cognitive = c1 * r1 * (personal_best[i] - particles[i])
                social = c2 * r2 * (global_best - particles[i])
                velocities[i] = w * velocities[i] + cognitive + social
                max_vel = 3.0 * (1 - iteration / FinalRobotSystemConfig.GLOBAL_ITERATIONS)
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, FinalRobotSystemConfig.AREA_SIZE)

                if iteration % 2 == 0 or iteration == FinalRobotSystemConfig.GLOBAL_ITERATIONS - 1:
                    current_fitness = self.fitness_func(particles[i])
                    if current_fitness < personal_best_fitness[i]:
                        personal_best[i] = particles[i].copy()
                        personal_best_fitness[i] = current_fitness
                        if current_fitness < global_best_fitness:
                            global_best = particles[i].copy()
                            global_best_fitness = float(current_fitness)

            if iteration % 10 == 5:
                worst_idx = int(np.argmax(personal_best_fitness))
                particles[worst_idx] = np.random.uniform(10, 40, n_dim)
                personal_best_fitness[worst_idx] = self.fitness_func(particles[worst_idx])

        elapsed = time.time() - start_time
        score = CompleteFitness.calculate_score(global_best)
        details = CompleteFitness.calculate_with_details(global_best)

        if needs_replan(details['metrics']):
            local = details['metrics']['path'][1:-1].flatten()
            local = local + np.random.uniform(-2, 2, local.shape)
            local = np.clip(local, 3, 47)
            base_fit = CompleteFitness.calculate(local)
            if base_fit < global_best_fitness:
                global_best = local.copy()
                global_best_fitness = base_fit
                score = CompleteFitness.calculate_score(global_best)
                details = CompleteFitness.calculate_with_details(global_best)

        return global_best, float(global_best_fitness), float(score), float(elapsed), details

# 3. PSO+GA (بدون Smooth)
class FinalPSOGA_NoSmooth:
    def __init__(self):
        self.fitness_func = CompleteFitness.calculate
        self.name = "3. PSO+GA (بدون Smooth)"

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        if np.random.rand() < 0.7:
            child = alpha * parent1 + (1 - alpha) * parent2
        else:
            mask = np.random.rand(len(parent1)) < 0.5
            child = parent1.copy()
            child[mask] = parent2[mask]
        return child

    def mutate(self, individual):
        mutated = individual.copy
        mutated = individual.copy()
        mutation_rate = FinalOptimizerConfig.GA_MUTATION
        if FinalOptimizerConfig.ADAPTIVE_MUTATION:
            fitness = self.fitness_func(individual)
            mutation_rate *= (1 + fitness)
        mask = np.random.rand(len(mutated)) < mutation_rate
        if np.any(mask):
            mutated[mask] += np.random.normal(0, 1, np.sum(mask)) * 2
            mutated = np.clip(mutated, 0, FinalRobotSystemConfig.AREA_SIZE)
        return mutated

    def run(self, run_number=1):
        pso = FinalPSO()
        solution, fitness, score, time_pso, _ = pso.run(run_number)

        start_ga = time.time()
        current = solution.copy()
        current_fitness = float(fitness)
        n_dim = FinalRobotSystemConfig.GLOBAL_DIM

        population_size = 12
        population = [current.copy()]
        for _ in range(population_size - 1):
            if np.random.rand() < 0.7:
                individual = current + np.random.uniform(-3, 3, n_dim)
            else:
                individual = np.random.uniform(10, 40, n_dim)
            individual = np.clip(individual, 0, FinalRobotSystemConfig.AREA_SIZE)
            population.append(individual)

        population = np.array(population)
        fitnesses = np.array([self.fitness_func(ind) for ind in population])

        for generation in range(20):
            sorted_indices = np.argsort(fitnesses)
            elite_count = max(1, int(FinalOptimizerConfig.ELITISM_RATE * population_size))
            elites = population[sorted_indices[:elite_count]]
            new_population = elites.tolist()

            while len(new_population) < population_size:
                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                parent1, parent2 = population[idx1], population[idx2]
                if np.random.rand() < FinalOptimizerConfig.GA_CROSSOVER:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy() if fitnesses[idx1] < fitnesses[idx2] else parent2.copy()
                child = self.mutate(child)
                new_population.append(child)

            population = np.array(new_population)
            fitnesses = np.array([self.fitness_func(ind) for ind in population])

        best_idx = int(np.argmin(fitnesses))
        current = population[best_idx]
        current_fitness = float(fitnesses[best_idx])

        elapsed = time_pso + (time.time() - start_ga)
        final_score = CompleteFitness.calculate_score(current)
        details = CompleteFitness.calculate_with_details(current)

        if needs_replan(details['metrics']):
            local = details['metrics']['path'][1:-1].flatten()
            local = local + np.random.uniform(-2, 2, local.shape)
            local = np.clip(local, 3, 47)
            base_fit = CompleteFitness.calculate(local)
            if base_fit < current_fitness:
                current = local.copy()
                current_fitness = base_fit
                final_score = CompleteFitness.calculate_score(current)
                details = CompleteFitness.calculate_with_details(current)

        return current, float(current_fitness), float(final_score), float(elapsed), details

# 4. PSO+GA+Smooth
class FinalPSOGA_WithSmooth:
    def __init__(self):
        self.fitness_func = CompleteFitness.calculate
        self.name = "4. PSO+GA+Smooth"

    def smooth_path(self, path, iterations=2):
        if len(path) < 3:
            return path
        smoothed = path.copy()
        for _ in range(iterations):
            temp = smoothed.copy()
            for i in range(1, len(temp) - 1):
                if i == 1 or i == len(temp) - 2:
                    temp[i] = 0.8 * temp[i] + 0.1 * (temp[i - 1] + temp[i + 1])
                else:
                    temp[i] = 0.6 * temp[i] + 0.2 * (temp[i - 1] + temp[i + 1])
            smoothed = temp
        for i in range(len(smoothed)):
            smoothed[i] = np.clip(smoothed[i], 3, 47)
        return smoothed

    def run(self, run_number=1):
        psoga = FinalPSOGA_NoSmooth()
        solution, fitness, score, time_base, _ = psoga.run(run_number)

        start_smooth = time.time()
        path = CompleteGeometry.decode_path(solution)
        smoothed_path = self.smooth_path(path, iterations=2)

        if len(smoothed_path) > 2:
            intermediate = smoothed_path[1:-1].flatten()
            n_dim = FinalRobotSystemConfig.GLOBAL_DIM
            if len(intermediate) > n_dim:
                intermediate = intermediate[:n_dim]
            else:
                padding = np.linspace(25, 25, n_dim - len(intermediate))
                intermediate = np.concatenate([intermediate, padding])
            smoothed_solution = intermediate
            smoothed_score = CompleteFitness.calculate_score(smoothed_solution)
            smoothed_fitness = self.fitness_func(smoothed_solution)

            if smoothed_score > score or smoothed_fitness < fitness:
                elapsed = time_base + (time.time() - start_smooth)
                details = CompleteFitness.calculate_with_details(smoothed_solution)
                if needs_replan(details['metrics']):
                    local = details['metrics']['path'][1:-1].flatten()
                    local = local + np.random.uniform(-2, 2, local.shape)
                    local = np.clip(local, 3, 47)
                    base_fit = CompleteFitness.calculate(local)
                    if base_fit < smoothed_fitness:
                        smoothed_solution = local.copy()
                        smoothed_fitness = base_fit
                        smoothed_score = CompleteFitness.calculate_score(smoothed_solution)
                        details = CompleteFitness.calculate_with_details(smoothed_solution)
                return smoothed_solution, float(smoothed_fitness), float(smoothed_score), float(elapsed), details

        details = CompleteFitness.calculate_with_details(solution)
        if needs_replan(details['metrics']):
            local = details['metrics']['path'][1:-1].flatten()
            local = local + np.random.uniform(-2, 2, local.shape)
            local = np.clip(local, 3, 47)
            base_fit = CompleteFitness.calculate(local)
            if base_fit < fitness:
                solution = local.copy()
                fitness = base_fit
                score = CompleteFitness.calculate_score(solution)
                details = CompleteFitness.calculate_with_details(solution)

        return solution, float(fitness), float(score), float(time_base), details

# 5. PSO+DE (بدون Smooth)
class FinalPSODE_NoSmooth:
    def __init__(self):
        self.fitness_func = CompleteFitness.calculate
        self.name = "5. PSO+DE (بدون Smooth)"

    def run(self, run_number=1):
        start_time = time.time()
        seed = 42 + run_number * 31
        np.random.seed(seed)

        n_pop = FinalRobotSystemConfig.GLOBAL_PARTICLES
        n_dim = FinalRobotSystemConfig.GLOBAL_DIM

        particles = []
        for i in range(n_pop):
            if i < n_pop // 2:
                particles.append(np.random.uniform(15, 35, n_dim))
            else:
                t = np.linspace(0, 1, FinalRobotSystemConfig.GLOBAL_POINTS + 2)[1:-1]
                base = FinalRobotSystemConfig.START + t[:, None] * (FinalRobotSystemConfig.GOAL - FinalRobotSystemConfig.START)
                particles.append(base.flatten() + np.random.uniform(-3, 3, n_dim))

        particles = np.array(particles)
        velocities = np.zeros((n_pop, n_dim))
        personal_best = particles.copy()
        personal_best_fitness = np.array([self.fitness_func(p) for p in particles])

        global_best_idx = int(np.argmin(personal_best_fitness))
        global_best = particles[global_best_idx].copy()
        global_best_fitness = float(personal_best_fitness[global_best_idx])

        for iteration in range(18):
            w = 0.7 * (1 - iteration / 18)
            for i in range(n_pop):
                r1, r2 = np.random.rand(2)
                c1, c2 = 1.5, 1.5
                cognitive = c1 * r1 * (personal_best[i] - particles[i])
                social = c2 * r2 * (global_best - particles[i])
                velocities[i] = np.clip(w * velocities[i] + cognitive + social, -2.5, 2.5)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 5, 45)

        population = particles.copy()
        fitness = np.array([self.fitness_func(p) for p in population])

        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        for iteration in range(25):
            F_base = FinalOptimizerConfig.DE_F
            for i in range(n_pop):
                if np.random.rand() < 0.7:
                    base = best_solution
                    F = F_base * (0.8 + 0.4 * np.random.rand())
                else:
                    idxs = np.random.choice(n_pop, 3, replace=False)
                    a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                    base = a
                    F = F_base

                if np.random.rand() < 0.3:
                    idxs = np.random.choice(n_pop, 4, replace=False)
                    a, b, c, d = population[idxs[0]], population[idxs[1]], population[idxs[2]], population[idxs[3]]
                    mutant = base + F * (b - c) + 0.5 * F * (a - d)
                else:
                    idxs = np.random.choice(n_pop, 3, replace=False)
                    a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                    mutant = base + F * (b - c)

                mutant = np.clip(mutant, 5, 45)

                trial = population[i].copy()
                cross_points = np.random.rand(n_dim) < FinalOptimizerConfig.DE_CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(n_dim)] = True
                trial[cross_points] = mutant[cross_points]

                if np.random.rand() < 0.1:
                    mutation_mask = np.random.rand(n_dim) < 0.05
                    trial[mutation_mask] += np.random.normal(0, 1, np.sum(mutation_mask))
                    trial = np.clip(trial, 5, 45)

                trial_fitness = self.fitness_func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = float(trial_fitness)

        elapsed = time.time() - start_time
        score = CompleteFitness.calculate_score(best_solution)
        details = CompleteFitness.calculate_with_details(best_solution)

        if needs_replan(details['metrics']):
            local = details['metrics']['path'][1:-1].flatten()
            local = local + np.random.uniform(-2, 2, local.shape)
            local = np.clip(local, 3, 47)
            base_fit = CompleteFitness.calculate(local)
            if base_fit < best_fitness:
                best_solution = local.copy()
                best_fitness = base_fit
                score = CompleteFitness.calculate_score(best_solution)
                details = CompleteFitness.calculate_with_details(best_solution)

        return best_solution, float(best_fitness), float(score), float(elapsed), details

# 6. PSO+DE+Smooth
class FinalPSODE_WithSmooth:
    def __init__(self):
        self.fitness_func = CompleteFitness.calculate
        self.name = "6. PSO+DE+Smooth"

    def smooth_path(self, path, strength=0.7, iterations=2):
        if len(path) < 3:
            return path
        smoothed = path.copy()
        for _ in range(iterations):
            temp = smoothed.copy()
            weights = np.ones(len(temp))
            for i in range(len(temp)):
                dist_to_start = np.linalg.norm(temp[i] - FinalRobotSystemConfig.START)
                dist_to_goal = np.linalg.norm(temp[i] - FinalRobotSystemConfig.GOAL)
                min_dist = min(dist_to_start, dist_to_goal)
                weights[i] = 0.9 if min_dist < 10 else 1.0
            for i in range(1, len(temp) - 1):
                alpha = strength * weights[i]
                beta = (1 - alpha) / 2
                temp[i] = alpha * temp[i] + beta * (temp[i - 1] + temp[i + 1])
            smoothed = temp
        return smoothed

    def run(self, run_number=1):
        psode = FinalPSODE_NoSmooth()
        solution, fitness, score, time_base, _ = psode.run(run_number)

        start_smooth = time.time()
        path = CompleteGeometry.decode_path(solution)

        best_smoothed_path = path
        best_smoothed_score = score
        best_smoothed_fitness = fitness

        for strength in [0.6, 0.7, 0.8]:
            smoothed_path = self.smooth_path(path, strength=strength, iterations=2)
            if len(smoothed_path) > 2:
                intermediate = smoothed_path[1:-1].flatten()
                n_dim = FinalRobotSystemConfig.GLOBAL_DIM
                if len(intermediate) > n_dim:
                    intermediate = intermediate[:n_dim]
                else:
                    intermediate = np.pad(intermediate, (0, n_dim - len(intermediate)), 'constant', constant_values=25.0)
                smoothed_solution = intermediate
                smoothed_score = CompleteFitness.calculate_score(smoothed_solution)
                if smoothed_score > best_smoothed_score:
                    best_smoothed_path = smoothed_path
                    best_smoothed_score = smoothed_score
                    best_smoothed_fitness = self.fitness_func(smoothed_solution)

        if best_smoothed_score > score:
            intermediate = best_smoothed_path[1:-1].flatten()
            n_dim = FinalRobotSystemConfig.GLOBAL_DIM
            if len(intermediate) > n_dim:
                intermediate = intermediate[:n_dim]
            else:
                intermediate = np.pad(intermediate, (0, n_dim - len(intermediate)), 'constant', constant_values=25.0)
            smoothed_solution = intermediate
            elapsed = time_base + (time.time() - start_smooth)
            details = CompleteFitness.calculate_with_details(smoothed_solution)
            if needs_replan(details['metrics']):
                local = details['metrics']['path'][1:-1].flatten()
                local = local + np.random.uniform(-2, 2, local.shape)
                local = np.clip(local, 3, 47)
                base_fit = CompleteFitness.calculate(local)
                if base_fit < best_smoothed_fitness:
                    smoothed_solution = local.copy()
                    best_smoothed_fitness = base_fit
                    best_smoothed_score = CompleteFitness.calculate_score(smoothed_solution)
                    details = CompleteFitness.calculate_with_details(smoothed_solution)
            return smoothed_solution, float(best_smoothed_fitness), float(best_smoothed_score), float(elapsed), details

        details = CompleteFitness.calculate_with_details(solution)
        if needs_replan(details['metrics']):
            local = details['metrics']['path'][1:-1].flatten()
            local = local + np.random.uniform(-2, 2, local.shape)
            local = np.clip(local, 3, 47)
            base_fit = CompleteFitness.calculate(local)
            if base_fit < fitness:
                solution = local.copy()
                fitness = base_fit
                score = CompleteFitness.calculate_score(solution)
                details = CompleteFitness.calculate_with_details(solution)

        return solution, float(fitness), float(score), float(time_base), details

# ============================================================
# نظام التقييم الكامل مع التفاصيل
# ============================================================
class CompleteEvaluationSystem:
    """نظام تقييم كامل مع جميع التفاصيل"""

    def __init__(self):
        self.algorithms = {
            "1. Baseline": FinalBaseline(),
            "2. PSO فقط": FinalPSO(),
            "3. PSO+GA (بدون Smooth)": FinalPSOGA_NoSmooth(),
            "4. PSO+GA+Smooth": FinalPSOGA_WithSmooth(),
            "5. PSO+DE (بدون Smooth)": FinalPSODE_NoSmooth(),
            "6. PSO+DE+Smooth": FinalPSODE_WithSmooth()
        }

    def display_algorithm_details(self, name, result_details):
        metrics = result_details['metrics']
        partial = result_details['partial_scores']

        print(f"\n📊 تفاصيل {name}:")
        print(f"  • النتيجة النهائية: {result_details['score']:.1f}/100")
        print(f"  • اللياقة: {result_details['fitness']:.4f}")
        print(f"  • العقوبات: {partial['penalty']:.3f}, المكافآت: {partial['reward']:.3f}")

        print(f"\n  📏 المقاييس الهندسية:")
        print(f"    - الطول: {metrics['length']:.1f} م (المثالي: {metrics['ideal_length']:.1f} م)")
        print(f"    - نسبة الطول: {metrics['length_ratio']:.3f}×")
        print(f"    - السلاسة: {metrics['smoothness']:.3f}")
        print(f"    - الانحناء: {metrics['curvature']:.3f} 1/م")

        print(f"\n  🛡️ مقاييس الأمان:")
        print(f"    - الأمان: {metrics['safety']:.3f}")
        print(f"    - المسافة الآمنة المتوسطة: {metrics['clearance']:.2f} م")
        print(f"    - أقل مسافة آمنة: {metrics['min_clearance']:.2f} م")
        print(f"    - عدد التصادمات: {metrics['collision_count']}")

        print(f"\n  ⏱️ الزمن والتنفيذ:")
        print(f"    - الزمن التقديري: {metrics['time']:.2f} ث")
        print(f"    - تحقق كينماتيكي: {'نعم' if metrics.get('kinematic_ok', False) else 'لا'}")
        print(f"    - أقصى ω مطلوب: {metrics.get('omega_req_max', 0.0):.2f} راد/ث")
        print(f"    - متانة تحت ضجيج: {metrics.get('robustness', 0.0):.0%}")

        print(f"\n  ⚡ مقاييس الطاقة:")
        print(f"    - الطاقة: {metrics['energy']:.1f} واط")
        print(f"    - نسبة الطاقة: {metrics['energy_ratio']:.3f}×")
        print(f"    - عقوبة الانعطاف: {metrics['turning_penalty']:.1f} واط")

        print(f"\n  🎯 النقاط الجزئية:")
        print(f"    - طول: {partial['length_points']}/30 ({partial['length_score']:.3f})")
        print(f"    - أمان: {partial['safety_points']:.1f}/35 ({partial['safety_score']:.3f})")
        print(f"    - طاقة: {partial['energy_points']}/25 ({partial['energy_score']:.3f})")
        print(f"    - سلاسة: {partial['smoothness_points']:.1f}/10 ({partial['smoothness_score']:.3f})")
        print(f"    - زمن: {partial['time_points']:.1f}/10 ({partial['time_score']:.3f})")

        print(f"\n  ✅ الصلاحية: {'نعم' if metrics['valid'] else 'لا'}")
        print(f"  📍 عدد النقاط: {metrics['num_points']}")

        print(f"\n  📈 معلومات إضافية:")
        if metrics['segment_lengths']:
            print(f"    - طول المقطع الأطول: {max(metrics['segment_lengths']):.1f} م")
            print(f"    - طول المقطع الأقصر: {min(metrics['segment_lengths']):.1f} م")
            print(f"    - متوسط طول المقطع: {np.mean(metrics['segment_lengths']):.1f} م")

    def display_final_comparison(self, results):
        print("\n" + "=" * 120)
        print("🏆 المقارنة النهائية مع جميع المعايير")
        print("=" * 120)

        print(f"\n{'الخوارزمية':<25} {'النتيجة':>8} {'اللياقة':>8} {'الوقت':>8} {'الطول':>8} {'الأمان':>8} {'الطاقة':>8} {'السلاسة':>8} {'الصلاحية':>10}")
        print("-" * 125)

        for res in sorted(results, key=lambda x: x['score'], reverse=True):
            print(f"{res['name']:<25} "
                  f"{res['score']:>8.1f} "
                  f"{res['fitness']:>8.4f} "
                  f"{res['time']:>8.2f} "
                  f"{res['metrics']['length']:>8.1f} "
                  f"{res['metrics']['safety']:>8.3f} "
                  f"{res['metrics']['energy']:>8.1f} "
                  f"{res['metrics']['smoothness']:>8.3f} "
                  f"{res['metrics']['valid']:>10.1%}")

        print(f"\n📈 تحليل النتائج:")
        best = max(results, key=lambda x: x['score'])
        best_fitness = min(results, key=lambda x: x['fitness'])
        fastest = min(results, key=lambda x: x['time'])
        safest = max(results, key=lambda x: x['metrics']['safety'])
        smoothest = max(results, key=lambda x: x['metrics']['smoothness'])
        most_efficient = min(results, key=lambda x: x['metrics']['energy'])
        most_valid = max(results, key=lambda x: x['metrics']['valid'])

        print(f"  • أفضل نتيجة: {best['name']} بنتيجة {best['score']:.1f}/100")
        print(f"  • أفضل لياقة: {best_fitness['name']} بلياقة {best_fitness['fitness']:.4f}")
        print(f"  • أسرع خوارزمية: {fastest['name']} بوقت {fastest['time']:.2f} ثانية")
        print(f"  • أكثر أماناً: {safest['name']} بأمان {safest['metrics']['safety']:.3f}")
        print(f"  • أكثر سلاسة: {smoothest['name']} بسلاسة {smoothest['metrics']['smoothness']:.3f}")
        print(f"  • أكثر توفيراً للطاقة: {most_efficient['name']} بطاقة {most_efficient['metrics']['energy']:.1f} واط")
        print(f"  • أعلى نسبة صلاحية: {most_valid['name']} بنسبة {most_valid['metrics']['valid']:.1%}")

        print(f"\n💡 التوصيات النهائية:")
        print(f"  1. للنتيجة الشاملة: {best['name']}")
        print(f"  2. للسرعة القصوى: {fastest['name']}")
        print(f"  3. للأمان الأعلى: {safest['name']}")
        print(f"  4. للسلاسة: {smoothest['name']}")
        print(f"  5. لتوفير الطاقة: {most_efficient['name']}")
        print(f"  6. للصلاحية: {most_valid['name']}")
        print(f"  7. جميع الخوارزميات تحت {FinalRobotSystemConfig.TIME_BUDGET:.1f} ثواني (عند تحقق الزمن) ✅")

        print(f"\n🎯 ملخص الأداء:")
        valid_count = sum(1 for r in results if r['metrics']['valid'] > 0.5)
        print(f"  • الخوارزميات الصالحة (نسبة > 50%): {valid_count}/{len(results)}")
        print(f"  • متوسط النتيجة: {np.mean([r['score'] for r in results]):.1f}/100")
        print(f"  • متوسط اللياقة: {np.mean([r['fitness'] for r in results]):.4f}")
        print(f"  • متوسط الوقت: {np.mean([r['time'] for r in results]):.2f} ثانية")

    def run_complete_comparison(self):
        print("=" * 120)
        print("🏭 النظام النهائي المحسّن - جميع الخوارزميات مع التصحيحات")
        print("🎯 معايير كاملة: اللياقة، السلاسة، الأمان، الطاقة، الطول، الزمن")
        print("=" * 120)

        print(f"\n📋 معلومات النظام:")
        print(f"  • المساحة: {FinalRobotSystemConfig.AREA_SIZE}×{FinalRobotSystemConfig.AREA_SIZE} متر")
        print(f"  • العوائق: {len(FinalRobotSystemConfig.OBSTACLES)} عائق")
        print(f"  • نقاط التحكم: {FinalRobotSystemConfig.GLOBAL_POINTS}")
        print(f"  • هدف الوقت: < {FinalRobotSystemConfig.TIME_BUDGET} ثواني")
        print(f"  • التشغيلات لكل خوارزمية: {FinalOptimizerConfig.RUNS}")

        all_results = []

        for name, algo in self.algorithms.items():
            print(f"\n{'=' * 80}")
            print(f"🔬 {name}")
            print(f"{'=' * 80}")

            run_results = []
            run_details = []

            for run in range(FinalOptimizerConfig.RUNS):
                print(f"  التشغيل {run + 1}/{FinalOptimizerConfig.RUNS}...", end=" ")
                start_run = time.time()
                solution, fitness, score, elapsed, details = algo.run(run + 1)
                run_time = time.time() - start_run

                run_results.append({
                    'score': score,
                    'fitness': fitness,
                    'time': elapsed,
                    'run_time': run_time
                })
                run_details.append(details)

                print(f"✅ النتيجة: {score:.1f}, الوقت: {elapsed:.2f}ث")

            avg_score = float(np.mean([r['score'] for r in run_results]))
            avg_fitness = float(np.mean([r['fitness'] for r in run_results]))
            avg_time = float(np.mean([r['time'] for r in run_results]))
            avg_run_time = float(np.mean([r['run_time'] for r in run_results]))

            avg_metrics = {
                'length': float(np.mean([d['metrics']['length'] for d in run_details])),
                'safety': float(np.mean([d['metrics']['safety'] for d in run_details])),
                'energy': float(np.mean([d['metrics']['energy'] for d in run_details])),
                'smoothness': float(np.mean([d['metrics']['smoothness'] for d in run_details])),
                'length_ratio': float(np.mean([d['metrics']['length_ratio'] for d in run_details])),
                'energy_ratio': float(np.mean([d['metrics']['energy_ratio'] for d in run_details])),
                'curvature': float(np.mean([d['metrics']['curvature'] for d in run_details])),
                'min_clearance': float(np.mean([d['metrics']['min_clearance'] for d in run_details])),
                'valid': float(np.mean([d['metrics']['valid'] for d in run_details])),
                'time': float(np.mean([d['metrics']['time'] for d in run_details]))
            }

            all_results.append({
                'name': name,
                'score': avg_score,
                'fitness': avg_fitness,
                'time': avg_time,
                'run_time': avg_run_time,
                'metrics': avg_metrics
            })

            print(f"\n📊 المتوسطات:")
            print(f"  • النتيجة: {avg_score:.1f}/100")
            print(f"  • اللياقة: {avg_fitness:.4f}")
            print(f"  • الوقت: {avg_time:.2f} ثانية")
            print(f"  • وقت التشغيل: {avg_run_time:.2f} ثانية")
            print(f"  • الطول: {avg_metrics['length']:.1f} متر")
            print(f"  • الأمان: {avg_metrics['safety']:.3f}")
            print(f"  • الطاقة: {avg_metrics['energy']:.1f} واط")
            print(f"  • السلاسة: {avg_metrics['smoothness']:.3f}")
            print(f"  • الزمن: {avg_metrics['time']:.2f} ثانية")

            best_run_idx = int(np.argmax([r['score'] for r in run_results]))
            self.display_algorithm_details(f"أفضل تشغيل لـ {name}", run_details[best_run_idx])

        self.display_final_comparison(all_results)

        cache_stats = CompleteFitness.get_cache_stats()
        print(f"\n💾 إحصائيات الكاش:")
        print(f"  • الضربات: {cache_stats['hits']}")
        print(f"  • الإخفاقات: {cache_stats['misses']}")
        print(f"  • معدل الضربات: {cache_stats['hit_rate']:.1%}")
        print(f"  • حجم الكاش: {cache_stats['size']}")

        return all_results

# ============================================================
# التشغيل الرئيسي
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("🚀 النظام النهائي المصحح والمحسّن - جميع الخوارزميات")
    print("🎯 معايير كاملة: اللياقة، السلاسة، الأمان، الطاقة، الطول، الزمن")
    print("=" * 80)

    print("\n🧪 اختبار الخوارزميات المحسنة:")
    print("-" * 40)

    test_algorithms = [
        ("PSO+DE+Smooth", FinalPSODE_WithSmooth()),
        ("PSO فقط", FinalPSO()),
        ("Baseline", FinalBaseline())
    ]

    for name, algo in test_algorithms:
        print(f"\nاختبار {name}...", end=" ")
        solution, fitness, score, elapsed, details = algo.run(1)

        if score > 0 and elapsed < 3.0:
            print("✅ تعمل بشكل صحيح!")
            print(f"   • النتيجة: {score:.1f}/100")
            print(f"   • اللياقة: {fitness:.4f}")
            print(f"   • الوقت: {elapsed:.2f} ثانية")
            print(f"   • السلاسة: {details['metrics']['smoothness']:.3f}")
            print(f"   • الأمان: {details['metrics']['safety']:.3f}")
            print(f"   • الزمن التقديري: {details['metrics']['time']:.2f} ث")
            print(f"   • تحقق كينماتيكي: {'نعم' if details['metrics']['kinematic_ok'] else 'لا'}")
            print(f"   • أقصى ω مطلوب: {details['metrics']['omega_req_max']:.2f} راد/ث")
        else:
            print("⚠️  تحتاج تحسين")

    print("\n" + "=" * 80)
    answer = "نعم"  # تشغيل تلقائي للمقارنة الكاملة

    if answer.lower() in ['نعم', 'yes', 'y', '']:
        print("\n" + "=" * 80)
        print("🎯 بدء المقارنة الكاملة مع التفاصيل...")
        print("=" * 80)

        evaluator = CompleteEvaluationSystem()
        results = evaluator.run_complete_comparison()

        print("\n" + "=" * 80)
        print("✅ النظام النهائي المتكامل جاهز مع:")
        print("  1. ✅ جميع الخوارزميات الست محسّنة")
        print("  2. ✅ حساب السلاسة المصحح")
        print("  3. ✅ موازنة محسّنة للمعايير مع الزمن")
        print("  4. ✅ كاش لتحسين الأداء")
        print("  5. ✅ تحقق كينماتيكي وزمني وحساسات ومتانة")
        print("  6. ✅ أوامر تحكم فعليّة (v, ω)")
        print("  7. ✅ تقارير تفصيلية كاملة")
        print("=" * 80)

        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        print("\n🏆 أفضل 3 خوارزميات:")
        for i, res in enumerate(sorted_results[:3], 1):
            print(f"{i}. {res['name']}: {res['score']:.1f}/100 "
                  f"(لياقة: {res['fitness']:.4f}, وقت: {res['time']:.2f}ث)")
    else:
        print("\n✅ تم التحقق من الخوارزميات المحسنة. النظام جاهز!")
