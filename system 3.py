"""
🚀 نظام تحسين مسار - الإصدار الشامل
🎯 مقارنة PSO+GA vs PSO+DE مع وبدون Smoothing
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# إعدادات النظام
# ============================================================
class ComprehensiveConfig:
    START = np.array([0.0, 0.0])
    GOAL  = np.array([20.0, 20.0])
    BOUNDS = (0.0, 20.0)
    
    DIM_POINTS = 5
    DIM = 10
    
    OBSTACLES = [
        {"center": np.array([7.0, 7.0]), "radius": 4.0},
        {"center": np.array([13.0, 13.0]), "radius": 4.0},
        {"center": np.array([5.0, 15.0]), "radius": 2.5},
        {"center": np.array([15.0, 5.0]), "radius": 2.5},
    ]
    
    IDEAL_LENGTH = 28.28
    MAX_LENGTH = 60.0

class ComprehensiveOptimizerConfig:
    N_PARTICLES = 15
    ITERATIONS = 40
    
    PSO_W = 0.5
    PSO_C1 = 1.5
    PSO_C2 = 1.5
    
    GA_CROSSOVER_RATE = 0.6
    GA_MUTATION_RATE = 0.2
    
    DE_F = 1.2
    DE_CR = 0.9
    
    RUNS = 5

# ============================================================
# هندسة النظام
# ============================================================
class ComprehensiveGeometry:
    
    @staticmethod
    def decode_path(solution):
        solution = np.asarray(solution).flatten()
        
        if len(solution) < ComprehensiveConfig.DIM:
            needed = ComprehensiveConfig.DIM - len(solution)
            t = np.linspace(0.1, 0.9, needed // 2 + 1)[:-1]
            extra = []
            for ti in t:
                if ti < 0.5:
                    x = 5 + np.sin(ti * 8 * np.pi) * 6
                    y = 5 + np.cos(ti * 8 * np.pi) * 6
                else:
                    x = 15 + np.sin(ti * 8 * np.pi) * 6
                    y = 15 + np.cos(ti * 8 * np.pi) * 6
                extra.extend([x, y])
            
            extra = extra[:needed]
            solution = np.concatenate([solution, extra])
        
        if len(solution) % 2 != 0:
            solution = np.append(solution, 10.0)
        
        points = solution.reshape(-1, 2)
        
        if len(points) < ComprehensiveConfig.DIM_POINTS:
            needed_points = ComprehensiveConfig.DIM_POINTS - len(points)
            extra_points = []
            for _ in range(needed_points):
                if np.random.rand() < 0.5:
                    x = np.random.uniform(3, 8)
                    y = np.random.uniform(12, 17)
                else:
                    x = np.random.uniform(12, 17)
                    y = np.random.uniform(3, 8)
                extra_points.append([x, y])
            
            extra_points = np.array(extra_points)
            points = np.vstack([points, extra_points])
        
        points = points[:ComprehensiveConfig.DIM_POINTS]
        points = np.clip(points, ComprehensiveConfig.BOUNDS[0], ComprehensiveConfig.BOUNDS[1])
        
        return np.vstack([ComprehensiveConfig.START, points, ComprehensiveConfig.GOAL])
    
    @staticmethod
    def calculate_path_metrics(path):
        if len(path) < 2:
            return {'length': 0, 'smoothness': 0.3, 'safety': 0.3, 'valid': False}
        
        length = 0.0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        
        angles = []
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0.1 and norm2 > 0.1:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)
        
        if angles:
            mean_angle = np.mean(angles)
            if mean_angle < np.deg2rad(20):
                smoothness = 0.9
            elif mean_angle < np.deg2rad(35):
                smoothness = 0.7
            elif mean_angle < np.deg2rad(50):
                smoothness = 0.5
            elif mean_angle < np.deg2rad(65):
                smoothness = 0.3
            else:
                smoothness = 0.1
        else:
            smoothness = 0.3
        
        safety_scores = []
        for obs in ComprehensiveConfig.OBSTACLES:
            center, radius = obs["center"], obs["radius"]
            for point in path:
                distance = np.linalg.norm(point - center)
                if distance <= radius:
                    safety_scores.append(0.0)
                elif distance < radius + 1.0:
                    safety_scores.append(0.3)
                elif distance < radius + 2.0:
                    safety_scores.append(0.6)
                elif distance < radius + 3.0:
                    safety_scores.append(0.8)
                else:
                    safety_scores.append(1.0)
        
        safety = np.mean(safety_scores) if safety_scores else 0.5
        valid = (safety > 0.2 and length < ComprehensiveConfig.MAX_LENGTH)
        
        return {
            'length': length,
            'smoothness': smoothness,
            'safety': safety,
            'valid': valid,
            'path': path
        }

# ============================================================
# دالة اللياقة
# ============================================================
class ComprehensiveFitness:
    
    @staticmethod
    def calculate(solution):
        path = ComprehensiveGeometry.decode_path(solution)
        metrics = ComprehensiveGeometry.calculate_path_metrics(path)
        
        if not metrics['valid']:
            return 0.7
        
        length = metrics['length']
        smoothness = metrics['smoothness']
        safety = metrics['safety']
        
        length_ratio = length / ComprehensiveConfig.IDEAL_LENGTH
        
        if length_ratio <= 1.1:
            norm_length = 0.8
        elif length_ratio <= 1.3:
            norm_length = 0.6
        elif length_ratio <= 1.5:
            norm_length = 0.4
        else:
            norm_length = 0.2
        
        norm_smoothness = smoothness * 0.3
        norm_safety = safety
        
        fitness = (
            0.25 * (1 - norm_length) +     
            0.30 * (1 - norm_smoothness) + 
            0.35 * (1 - norm_safety) +
            0.10 * (1 if smoothness < 0.3 else 0)
        )
        
        if safety < 0.3:
            fitness += 0.2
        if smoothness < 0.2:
            fitness += 0.1
        
        return max(0.0, min(1.0, fitness))
    
    @staticmethod
    def calculate_score(solution):
        path = ComprehensiveGeometry.decode_path(solution)
        metrics = ComprehensiveGeometry.calculate_path_metrics(path)
        
        if not metrics['valid']:
            return 0.0
        
        length = metrics['length']
        smoothness = metrics['smoothness']
        safety = metrics['safety']
        
        length_score = max(0, 100 - (length - ComprehensiveConfig.IDEAL_LENGTH) * 5)
        smoothness_score = smoothness * 100
        safety_score = safety * 100
        
        final_score = 0.5 * smoothness_score + 0.3 * safety_score + 0.2 * length_score
        
        return final_score

# ============================================================
# Smoothing ذكي
# ============================================================
class SmartSmoothing:
    
    @staticmethod
    def apply_smoothing(path, original_score=None):
        """Smoothing ذكي حسب جودة المسار"""
        if len(path) < 3:
            return path.copy()
        
        smoothed = path.copy()
        
        # تحديد قوة التنعيم حسب جودة المسار
        if original_score is not None and original_score > 80:
            smoothing_strength = 0.1  # تنعيم خفيف جداً للمسارات الجيدة
        elif original_score is not None and original_score > 50:
            smoothing_strength = 0.3  # تنعيم متوسط
        else:
            smoothing_strength = 0.5  # تنعيم قوي للمسارات السيئة
        
        # التنعيم الأساسي
        for i in range(1, len(smoothed) - 1):
            smoothed[i] = (1 - smoothing_strength) * smoothed[i] + \
                         (smoothing_strength / 2) * (smoothed[i-1] + smoothed[i+1])
        
        # معالجة الزوايا الحادة
        for i in range(1, len(smoothed) - 1):
            v1 = smoothed[i] - smoothed[i-1]
            v2 = smoothed[i+1] - smoothed[i]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0.1 and norm2 > 0.1:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                
                if angle > np.deg2rad(70):
                    midpoint = 0.5 * (smoothed[i-1] + smoothed[i+1])
                    smoothed[i] = 0.4 * smoothed[i] + 0.6 * midpoint
        
        smoothed = np.clip(smoothed, ComprehensiveConfig.BOUNDS[0], ComprehensiveConfig.BOUNDS[1])
        return smoothed

# ============================================================
# الخوارزميات الست
# ============================================================

# 1. Baseline
class BaselineAlgo:
    def __init__(self, fitness_func):
        self.fitness_func = fitness_func
    
    def run(self, run_number=1):
        np.random.seed(42 + run_number)
        
        n_dim = ComprehensiveConfig.DIM
        solution = []
        for i in range(n_dim // 2):
            x = np.random.uniform(5, 15)
            y = np.random.uniform(5, 15)
            solution.extend([x, y])
        
        solution = np.array(solution)
        solution = np.clip(solution, 0, 20)
        
        fitness = self.fitness_func(solution)
        score = ComprehensiveFitness.calculate_score(solution)
        
        return solution, fitness, score, 0.1

# 2. PSO فقط
class PSOOnlyAlgo:
    def __init__(self, fitness_func):
        self.fitness_func = fitness_func
    
    def run(self, run_number=1):
        np.random.seed(42 + run_number * 10)
        
        n_particles = ComprehensiveOptimizerConfig.N_PARTICLES
        n_dim = ComprehensiveConfig.DIM
        
        particles = np.random.uniform(5, 15, (n_particles, n_dim))
        velocities = np.zeros((n_particles, n_dim))
        
        personal_best = particles.copy()
        personal_best_fitness = np.array([self.fitness_func(p) for p in particles])
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = particles[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        start_time = time.time()
        
        for iteration in range(ComprehensiveOptimizerConfig.ITERATIONS):
            w = ComprehensiveOptimizerConfig.PSO_W
            c1 = ComprehensiveOptimizerConfig.PSO_C1
            c2 = ComprehensiveOptimizerConfig.PSO_C2
            
            for i in range(n_particles):
                r1, r2 = np.random.rand(2)
                cognitive = c1 * r1 * (personal_best[i] - particles[i])
                social = c2 * r2 * (global_best - particles[i])
                
                velocities[i] = w * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -1.0, 1.0)
                
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, 20)
                
                current_fitness = self.fitness_func(particles[i])
                
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = current_fitness
                    
                    if current_fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = current_fitness
        
        elapsed_time = time.time() - start_time
        score = ComprehensiveFitness.calculate_score(global_best)
        
        return global_best, global_best_fitness, score, elapsed_time

# 3. PSO+GA (بدون Smoothing)
class PSOGA_NoSmooth:
    def __init__(self, fitness_func):
        self.fitness_func = fitness_func
    
    def run(self, run_number=1):
        np.random.seed(42 + run_number * 10)
        
        n_particles = ComprehensiveOptimizerConfig.N_PARTICLES
        n_dim = ComprehensiveConfig.DIM
        
        particles = []
        for _ in range(n_particles):
            particle = []
            for j in range(n_dim // 2):
                if j % 2 == 0:
                    x = np.random.uniform(2, 8)
                    y = np.random.uniform(12, 18)
                else:
                    x = np.random.uniform(12, 18)
                    y = np.random.uniform(2, 8)
                
                x += np.random.uniform(-2, 2)
                y += np.random.uniform(-2, 2)
                particle.extend([x, y])
            
            particle = np.array(particle[:n_dim])
            particles.append(particle)
        
        particles = np.array(particles)
        velocities = np.zeros((n_particles, n_dim))
        
        personal_best = particles.copy()
        personal_best_fitness = np.array([self.fitness_func(p) for p in particles])
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = particles[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        start_time = time.time()
        
        for iteration in range(ComprehensiveOptimizerConfig.ITERATIONS):
            w = ComprehensiveOptimizerConfig.PSO_W
            c1 = ComprehensiveOptimizerConfig.PSO_C1
            c2 = ComprehensiveOptimizerConfig.PSO_C2
            
            for i in range(n_particles):
                r1, r2 = np.random.rand(2)
                cognitive = c1 * r1 * (personal_best[i] - particles[i])
                social = c2 * r2 * (global_best - particles[i])
                
                velocities[i] = w * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -0.8, 0.8)
                
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, 20)
                
                current_fitness = self.fitness_func(particles[i])
                
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = current_fitness
                    
                    if current_fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = current_fitness
            
            if iteration % 5 == 0 and iteration > 10:
                sorted_idx = np.argsort(personal_best_fitness)
                idx1, idx2 = sorted_idx[0], sorted_idx[1]
                p1, p2 = personal_best[idx1], personal_best[idx2]
                
                if np.random.rand() < ComprehensiveOptimizerConfig.GA_CROSSOVER_RATE:
                    alpha = np.random.rand()
                    child = alpha * p1 + (1 - alpha) * p2
                    
                    if np.random.rand() < ComprehensiveOptimizerConfig.GA_MUTATION_RATE:
                        mutation_mask = np.random.rand(n_dim) < 0.2
                        child += mutation_mask * np.random.uniform(-1, 1, n_dim)
                        child = np.clip(child, 0, 20)
                    
                    worst_idx = sorted_idx[-1]
                    particles[worst_idx] = child
                    personal_best[worst_idx] = child
                    personal_best_fitness[worst_idx] = self.fitness_func(child)
        
        elapsed_time = time.time() - start_time
        score = ComprehensiveFitness.calculate_score(global_best)
        
        return global_best, global_best_fitness, score, elapsed_time

# 4. PSO+DE (بدون Smoothing)
class PSODE_NoSmooth:
    def __init__(self, fitness_func):
        self.fitness_func = fitness_func
    
    def run(self, run_number=1):
        np.random.seed(42 + run_number * 10 + 1000)  # بذرة مختلفة
        
        n_particles = ComprehensiveOptimizerConfig.N_PARTICLES
        n_dim = ComprehensiveConfig.DIM
        
        population = []
        for i in range(n_particles):
            particle = []
            for j in range(n_dim // 2):
                ratio = (j + 1) / (n_dim // 2 + 1)
                x = ratio * 20
                y = (1 - ratio) * 20
                
                if i % 2 == 0:
                    x += np.random.uniform(-4, 4)
                    y += np.random.uniform(-4, 4)
                else:
                    x += np.sin(j * np.pi / 2) * 5
                    y += np.cos(j * np.pi / 2) * 5
                
                particle.extend([x, y])
            
            particle = np.array(particle[:n_dim])
            particle = np.clip(particle, 0, 20)
            population.append(particle)
        
        population = np.array(population)
        fitness_values = np.array([self.fitness_func(p) for p in population])
        
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        start_time = time.time()
        
        # DE أولاً (20 تكرار)
        for iteration in range(20):
            for i in range(n_particles):
                idxs = np.random.choice(n_particles, 3, replace=False)
                a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = best_solution + ComprehensiveOptimizerConfig.DE_F * (b - c)
                mutant = np.clip(mutant, 0, 20)
                
                trial = population[i].copy()
                cross_points = np.random.rand(n_dim) < ComprehensiveOptimizerConfig.DE_CR
                trial[cross_points] = mutant[cross_points]
                
                trial_fitness = self.fitness_func(trial)
                if trial_fitness < fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
        
        # PSO بعد ذلك (20 تكرار)
        velocities = np.zeros((n_particles, n_dim))
        personal_best = population.copy()
        personal_best_fitness = fitness_values.copy()
        
        for iteration in range(20):
            w = ComprehensiveOptimizerConfig.PSO_W * (1 - iteration/20 * 0.3)
            
            for i in range(n_particles):
                r1, r2 = np.random.rand(2)
                cognitive = ComprehensiveOptimizerConfig.PSO_C1 * r1 * (personal_best[i] - population[i])
                social = ComprehensiveOptimizerConfig.PSO_C2 * r2 * (best_solution - population[i])
                
                velocities[i] = w * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -1.0, 1.0)
                
                new_particle = population[i] + velocities[i]
                new_particle = np.clip(new_particle, 0, 20)
                
                new_fitness = self.fitness_func(new_particle)
                if new_fitness < personal_best_fitness[i]:
                    personal_best[i] = new_particle.copy()
                    personal_best_fitness[i] = new_fitness
                    
                    if new_fitness < best_fitness:
                        best_solution = new_particle.copy()
                        best_fitness = new_fitness
                
                population[i] = new_particle
                fitness_values[i] = new_fitness
        
        elapsed_time = time.time() - start_time
        score = ComprehensiveFitness.calculate_score(best_solution)
        
        return best_solution, best_fitness, score, elapsed_time

# 5. PSO+GA+Smoothing
class PSOGA_WithSmooth:
    def __init__(self, fitness_func):
        self.fitness_func = fitness_func
        self.base_algo = PSOGA_NoSmooth(fitness_func)
    
    def run(self, run_number=1):
        # الحصول على الحل الأساسي
        solution, base_fitness, base_score, base_time = self.base_algo.run(run_number)
        
        # تطبيق Smoothing ذكي
        path = ComprehensiveGeometry.decode_path(solution)
        smoothed_path = SmartSmoothing.apply_smoothing(path, base_score)
        
        if len(smoothed_path) > 2:
            intermediate_points = smoothed_path[1:-1]
            smoothed_solution = intermediate_points.flatten()
            
            if len(smoothed_solution) < ComprehensiveConfig.DIM:
                smoothed_solution = np.concatenate([
                    smoothed_solution,
                    np.random.uniform(5, 15, ComprehensiveConfig.DIM - len(smoothed_solution))
                ])
            
            smoothed_solution = smoothed_solution[:ComprehensiveConfig.DIM]
            
            smoothed_score = ComprehensiveFitness.calculate_score(smoothed_solution)
            
            return smoothed_solution, self.fitness_func(smoothed_solution), smoothed_score, base_time + 0.2
        else:
            return solution, base_fitness, base_score, base_time

# 6. PSO+DE+Smoothing
class PSODE_WithSmooth:
    def __init__(self, fitness_func):
        self.fitness_func = fitness_func
        self.base_algo = PSODE_NoSmooth(fitness_func)
    
    def run(self, run_number=1):
        # الحصول على الحل الأساسي
        solution, base_fitness, base_score, base_time = self.base_algo.run(run_number)
        
        # تطبيق Smoothing ذكي
        path = ComprehensiveGeometry.decode_path(solution)
        smoothed_path = SmartSmoothing.apply_smoothing(path, base_score)
        
        if len(smoothed_path) > 2:
            intermediate_points = smoothed_path[1:-1]
            smoothed_solution = intermediate_points.flatten()
            
            if len(smoothed_solution) < ComprehensiveConfig.DIM:
                smoothed_solution = np.concatenate([
                    smoothed_solution,
                    np.random.uniform(5, 15, ComprehensiveConfig.DIM - len(smoothed_solution))
                ])
            
            smoothed_solution = smoothed_solution[:ComprehensiveConfig.DIM]
            
            smoothed_score = ComprehensiveFitness.calculate_score(smoothed_solution)
            
            return smoothed_solution, self.fitness_func(smoothed_solution), smoothed_score, base_time + 0.2
        else:
            return solution, base_fitness, base_score, base_time

# ============================================================
# نظام المقارنة الشامل
# ============================================================
def run_comprehensive_comparison():
    """تشغيل المقارنة الشاملة مع وبدون Smoothing"""
    print("=" * 80)
    print("🔥 مقارنة شاملة - 6 خوارزميات (مع وبدون Smoothing)")
    print("🎯 لرؤية التفوق الحقيقي لـ PSO+DE vs PSO+GA")
    print("=" * 80)
    
    print(f"\n📊 الخوارزميات المدروسة:")
    print("  1. Baseline (بدون تحسين)")
    print("  2. PSO فقط")
    print("  3. PSO+GA (بدون Smoothing)")
    print("  4. PSO+DE (بدون Smoothing)")
    print("  5. PSO+GA+Smoothing")
    print("  6. PSO+DE+Smoothing")
    print("\n📈 Smoothing الذكي: يحدد قوة التنعيم حسب جودة المسار الأصلي")
    
    fitness_func = ComprehensiveFitness.calculate
    
    algorithms = {
        "Baseline": BaselineAlgo(fitness_func),
        "PSO فقط": PSOOnlyAlgo(fitness_func),
        "PSO+GA (بدون Smooth)": PSOGA_NoSmooth(fitness_func),
        "PSO+DE (بدون Smooth)": PSODE_NoSmooth(fitness_func),
        "PSO+GA+Smooth": PSOGA_WithSmooth(fitness_func),
        "PSO+DE+Smooth": PSODE_WithSmooth(fitness_func)
    }
    
    results = {}
    
    for name, algo in algorithms.items():
        print(f"\n{'='*60}")
        print(f"🔬 {name}")
        print(f"{'='*60}")
        
        all_scores = []
        all_fitness = []
        all_lengths = []
        all_smoothness = []
        all_safety = []
        all_times = []
        
        for run in range(ComprehensiveOptimizerConfig.RUNS):
            solution, fitness, score, exec_time = algo.run(run + 1)
            
            path = ComprehensiveGeometry.decode_path(solution)
            metrics = ComprehensiveGeometry.calculate_path_metrics(path)
            
            all_scores.append(score)
            all_fitness.append(fitness)
            all_lengths.append(metrics['length'])
            all_smoothness.append(metrics['smoothness'])
            all_safety.append(metrics['safety'])
            all_times.append(exec_time)
        
        avg_score = np.mean(all_scores)
        avg_fitness = np.mean(all_fitness)
        avg_length = np.mean(all_lengths)
        avg_smoothness = np.mean(all_smoothness)
        avg_safety = np.mean(all_safety)
        avg_time = np.mean(all_times)
        
        results[name] = {
            'score': avg_score,
            'fitness': avg_fitness,
            'length': avg_length,
            'smoothness': avg_smoothness,
            'safety': avg_safety,
            'time': avg_time
        }
        
        print(f"\n📊 النتائج النهائية:")
        print(f"  • النتيجة: {avg_score:.1f}/100")
        print(f"  • اللياقة: {avg_fitness:.4f}")
        print(f"  • السلاسة: {avg_smoothness:.3f}")
        print(f"  • الأمان: {avg_safety:.3f}")
        print(f"  • الطول: {avg_length:.2f} م")
        print(f"  • الوقت: {avg_time:.2f} ثانية")
    
    # المقارنة النهائية
    print("\n" + "="*80)
    print("🏆 المقارنة النهائية - جميع الخوارزميات")
    print("="*80)
    
    print(f"\n{'الخوارزمية':<25} {'النتيجة':>8} {'السلاسة':>10} {'الأمان':>10} {'الطول':>10} {'الوقت':>8}")
    print("-" * 83)
    
    for name, data in results.items():
        print(f"{name:<25} {data['score']:>8.1f} {data['smoothness']:>10.3f} "
              f"{data['safety']:>10.3f} {data['length']:>10.2f} {data['time']:>8.2f}")
    
    print("-" * 83)
    
    # تحليل مقارنة بدون Smoothing
    print(f"\n📊 المقارنة بدون Smoothing:")
    if "PSO+GA (بدون Smooth)" in results and "PSO+DE (بدون Smooth)" in results:
        ga_no = results["PSO+GA (بدون Smooth)"]
        de_no = results["PSO+DE (بدون Smooth)"]
        
        score_diff_no = de_no['score'] - ga_no['score']
        time_diff_no = de_no['time'] - ga_no['time']
        
        print(f"  • PSO+GA: {ga_no['score']:.1f}/100, الوقت: {ga_no['time']:.2f} ثانية")
        print(f"  • PSO+DE: {de_no['score']:.1f}/100, الوقت: {de_no['time']:.2f} ثانية")
        print(f"  • الفرق: {score_diff_no:+.1f} نقطة لصالح {'PSO+DE' if score_diff_no > 0 else 'PSO+GA'}")
        print(f"  • فرق الوقت: {time_diff_no:+.2f} ثانية")
        
        if abs(score_diff_no) < 5:
            print(f"  ⚖️ النتائج متقاربة بدون Smoothing")
        elif score_diff_no > 5:
            print(f"  ✅ PSO+DE يتفوق بدون Smoothing!")
        else:
            print(f"  ✅ PSO+GA يتفوق بدون Smoothing!")
    
    # تحليل مقارنة مع Smoothing
    print(f"\n📊 المقارنة مع Smoothing:")
    if "PSO+GA+Smooth" in results and "PSO+DE+Smooth" in results:
        ga_with = results["PSO+GA+Smooth"]
        de_with = results["PSO+DE+Smooth"]
        
        score_diff_with = de_with['score'] - ga_with['score']
        time_diff_with = de_with['time'] - ga_with['time']
        
        print(f"  • PSO+GA+Smooth: {ga_with['score']:.1f}/100, الوقت: {ga_with['time']:.2f} ثانية")
        print(f"  • PSO+DE+Smooth: {de_with['score']:.1f}/100, الوقت: {de_with['time']:.2f} ثانية")
        print(f"  • الفرق: {score_diff_with:+.1f} نقطة لصالح {'PSO+DE' if score_diff_with > 0 else 'PSO+GA'}")
        print(f"  • فرق الوقت: {time_diff_with:+.2f} ثانية")
        
        if abs(score_diff_with) < 5:
            print(f"  ⚖️ النتائج متقاربة مع Smoothing")
        elif score_diff_with > 5:
            print(f"  ✅ PSO+DE يتفوق مع Smoothing!")
        else:
            print(f"  ✅ PSO+GA يتفوق مع Smoothing!")
    
    # تحليل تأثير Smoothing
    print(f"\n📈 تأثير Smoothing:")
    if "PSO+GA (بدون Smooth)" in results and "PSO+GA+Smooth" in results:
        ga_improvement = results["PSO+GA+Smooth"]['score'] - results["PSO+GA (بدون Smooth)"]['score']
        ga_time_penalty = results["PSO+GA+Smooth"]['time'] - results["PSO+GA (بدون Smooth)"]['time']
        print(f"  • تحسن PSO+GA: {ga_improvement:+.1f} نقطة (+{ga_improvement/results['PSO+GA (بدون Smooth)']['score']*100:.1f}%)")
        print(f"  • تكلفة الوقت: {ga_time_penalty:+.2f} ثانية")
    
    if "PSO+DE (بدون Smooth)" in results and "PSO+DE+Smooth" in results:
        de_improvement = results["PSO+DE+Smooth"]['score'] - results["PSO+DE (بدون Smooth)"]['score']
        de_time_penalty = results["PSO+DE+Smooth"]['time'] - results["PSO+DE (بدون Smooth)"]['time']
        print(f"  • تحسن PSO+DE: {de_improvement:+.1f} نقطة (+{de_improvement/results['PSO+DE (بدون Smooth)']['score']*100:.1f}%)")
        print(f"  • تكلفة الوقت: {de_time_penalty:+.2f} ثانية")
    
    print("\n" + "="*80)
    print("💡 الخلاصة:")
    print("  هذه المقارنة ستظهر:")
    print("  1. أي خوارزمية أفضل بدون مساعدة خارجية")
    print("  2. تأثير Smoothing الذكي على كل خوارزمية")
    print("  3. هل يستحق PSO+DE التعقيد الإضافي؟")

# ============================================================
# التشغيل
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    run_comprehensive_comparison()
