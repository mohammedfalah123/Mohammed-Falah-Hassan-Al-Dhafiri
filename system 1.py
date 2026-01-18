"""
النظام النهائي المصحح تماماً - بدون أخطاء
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# البذرة الرئيسية
# ============================================================
MAIN_SEED = 42
np.random.seed(MAIN_SEED)

print("=" * 70)
print("🚀 نظام تحسين مسار الروبوت - الإصدار النهائي الآمن")
print(f"🔧 البذرة الرئيسية: {MAIN_SEED}")
print("=" * 70)

# ============================================================
# إعدادات بسيطة وآمنة
# ============================================================
class SafeConfig:
    """إعدادات آمنة تماماً"""
    
    # هندسة المشكلة
    DIM_POINTS = 5           # 5 نقاط وسيطة
    DIM = 10                 # 10 أبعاد مباشرة (بدون حساب)
    BOUNDS = (0.0, 10.0)
    
    # نقاط البداية والنهاية
    START = np.array([0.0, 0.0])
    GOAL  = np.array([10.0, 10.0])
    
    # إعدادات الخوارزميات
    N_PARTICLES = 15
    ITERATIONS = 30
    RUNS = 3                 # 3 تشغيلات فقط للسرعة
    
    # عوائق بسيطة
    OBSTACLES = [
        {"center": np.array([3.0, 3.0]), "radius": 1.5},
        {"center": np.array([7.0, 7.0]), "radius": 1.2},
    ]
    
    # معايير
    IDEAL_LENGTH = 14.14

# ============================================================
# دوال هندسية آمنة
# ============================================================
class SafeGeometry:
    """دوال هندسية بسيطة وآمنة"""
    
    @staticmethod
    def decode_path(solution):
        """تحويل آمن للحل إلى مسار"""
        # تأكد أن الحل هو متجه
        solution = np.asarray(solution).flatten()
        
        # إذا كان قصيراً، أكمله
        if len(solution) < SafeConfig.DIM:
            needed = SafeConfig.DIM - len(solution)
            extra = np.random.uniform(2, 8, needed)
            solution = np.concatenate([solution, extra])
        
        # تحويل إلى نقاط (تأكد من العدد الزوجي)
        if len(solution) % 2 != 0:
            solution = np.append(solution, 5.0)
        
        points = solution.reshape(-1, 2)
        return np.vstack([SafeConfig.START, points, SafeConfig.GOAL])
    
    @staticmethod
    def path_length(path):
        """طول المسار"""
        if len(path) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    
    @staticmethod
    def calculate_angles(path):
        """زوايا الانعطاف"""
        angles = []
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1e-9 and norm2 > 1e-9:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        return np.array(angles) if angles else np.array([])
    
    @staticmethod
    def smoothness_score(path):
        """درجة سلاسة آمنة"""
        angles = SafeGeometry.calculate_angles(path)
        if len(angles) == 0:
            return 0.8
        
        mean_angle = np.mean(angles)
        
        if mean_angle < np.deg2rad(30):
            return 0.9
        elif mean_angle < np.deg2rad(60):
            return 0.7
        else:
            return 0.5
    
    @staticmethod
    def safety_score(path):
        """درجة أمان"""
        if len(path) == 0:
            return 0.0
        
        min_distance = float('inf')
        
        for obs in SafeConfig.OBSTACLES:
            center, radius = obs["center"], obs["radius"]
            
            for point in path:
                distance = np.linalg.norm(point - center) - radius
                min_distance = min(min_distance, distance)
        
        if min_distance >= 0.5:
            return 1.0
        elif min_distance <= 0:
            return 0.0
        else:
            return min_distance / 0.5
    
    @staticmethod
    def calculate_energy(path):
        """حساب الطاقة"""
        length = SafeGeometry.path_length(path)
        return length * 5.0
    
    @staticmethod
    def simple_smoothing(path):
        """تنعيم بسيط وآمن"""
        if len(path) < 3:
            return path.copy()
        
        smoothed = path.copy()
        
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1e-9 and norm2 > 1e-9:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                
                if angle > np.deg2rad(60):
                    target = 0.5 * (path[i-1] + path[i+1])
                    smoothed[i] = 0.7 * path[i] + 0.3 * target
        
        return smoothed

# ============================================================
# دالة لياقة آمنة
# ============================================================
class SafeFitness:
    """دالة لياقة آمنة"""
    
    @staticmethod
    def calculate(solution):
        """حساب اللياقة"""
        path = SafeGeometry.decode_path(solution)
        
        # المقاييس الأساسية
        length = SafeGeometry.path_length(path)
        smoothness = SafeGeometry.smoothness_score(path)
        safety = SafeGeometry.safety_score(path)
        energy = SafeGeometry.calculate_energy(path)
        
        # تطبيع
        norm_length = max(0, 1 - (length - SafeConfig.IDEAL_LENGTH) / 10.0)
        norm_smoothness = smoothness
        norm_safety = safety
        norm_energy = max(0, 1 - energy / 150.0)
        
        # اللياقة النهائية
        fitness = (
            0.30 * (1 - norm_length) +
            0.30 * (1 - norm_smoothness) +
            0.25 * (1 - norm_safety) +
            0.15 * (1 - norm_energy)
        )
        
        return max(0.0, min(1.0, fitness))

# ============================================================
# خوارزمية PSO+GA آمنة
# ============================================================
class SafeHybridPSOGA:
    """خوارزمية PSO+GA آمنة تماماً"""
    
    def __init__(self, fitness_func):
        self.fitness_func = fitness_func
    
    def run(self, run_number=1):
        """تشغيل آمن للخوارزمية"""
        n_particles = SafeConfig.N_PARTICLES
        n_dim = SafeConfig.DIM
        bounds_min, bounds_max = SafeConfig.BOUNDS
        
        # 1. تهيئة بسيطة وآمنة
        particles = np.random.uniform(bounds_min + 1, bounds_max - 1, 
                                     (n_particles, n_dim))
        velocities = np.random.uniform(-0.5, 0.5, (n_particles, n_dim))
        
        personal_best = particles.copy()
        personal_best_scores = np.array([self.fitness_func(p) for p in particles])
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best = particles[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        start_time = time.perf_counter()
        
        # 2. حلقة التكرار الآمنة
        for iteration in range(SafeConfig.ITERATIONS):
            progress = iteration / SafeConfig.ITERATIONS
            
            # معاملات ثابتة
            w = 0.7
            c1, c2 = 1.5, 1.5
            
            # خطوة PSO
            for i in range(n_particles):
                current_fitness = self.fitness_func(particles[i])
                
                if current_fitness < personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = current_fitness
                
                if current_fitness < global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = current_fitness
                
                r1, r2 = np.random.rand(2)
                cognitive = c1 * r1 * (personal_best[i] - particles[i])
                social = c2 * r2 * (global_best - particles[i])
                
                velocities[i] = w * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -1.0, 1.0)
                
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], bounds_min, bounds_max)
            
            # خطوة GA بسيطة وآمنة (كل 5 تكرارات)
            if iteration % 5 == 0 and iteration > 10:
                # اختيار أفضل 3
                best_idx = np.argsort(personal_best_scores)[:3]
                parents = personal_best[best_idx]
                
                # توليد طفلين
                children = []
                for _ in range(2):
                    # اختيار أباء آمن
                    idx1, idx2 = np.random.choice(3, 2, replace=False)
                    p1, p2 = parents[idx1], parents[idx2]
                    
                    # تهجين آمن
                    alpha = np.random.rand()
                    child = alpha * p1 + (1 - alpha) * p2
                    
                    # طفرة آمنة
                    if np.random.rand() < 0.3:
                        # طفرة في نقطة واحدة فقط
                        mutation_point = np.random.randint(0, n_dim)
                        child[mutation_point] += np.random.uniform(-0.3, 0.3)
                    
                    child = np.clip(child, bounds_min, bounds_max)
                    children.append(child)
                
                # استبدال الأسوأ
                worst_idx = np.argsort(personal_best_scores)[-2:]
                for idx, child in zip(worst_idx, children):
                    particles[idx] = child
                    personal_best[idx] = child
                    personal_best_scores[idx] = self.fitness_func(child)
        
        elapsed_time = time.perf_counter() - start_time
        
        return global_best, global_best_score, elapsed_time

# ============================================================
# خوارزمية PSO+GA+Smoothing آمنة
# ============================================================
class SafePSOGASmoothing:
    """PSO+GA مع Smoothing آمن"""
    
    def __init__(self, fitness_func):
        self.fitness_func = fitness_func
        self.hybrid = SafeHybridPSOGA(fitness_func)
    
    def run(self, run_number=1):
        """تشغيل آمن مع Smoothing"""
        print(f"\n   🔄 PSO+GA+Smoothing (التشغيل {run_number})")
        
        # 1. تشغيل PSO+GA
        print("   📈 المرحلة 1: PSO+GA")
        solution, fitness, time_pso_ga = self.hybrid.run(run_number)
        print(f"   ✓ PSO+GA: اللياقة = {fitness:.4f}")
        
        # 2. تطبيق Smoothing
        print("   🎯 المرحلة 2: Smoothing")
        path = SafeGeometry.decode_path(solution)
        
        # تجربة Smoothing
        smoothed_path = SafeGeometry.simple_smoothing(path)
        
        # تحويل إلى حل
        intermediate_points = smoothed_path[1:-1]
        smoothed_solution = intermediate_points.flatten()
        smoothed_fitness = self.fitness_func(smoothed_solution)
        
        improvement = fitness - smoothed_fitness
        
        print(f"   📊 مقارنة:")
        print(f"     • قبل Smoothing: {fitness:.4f}")
        print(f"     • بعد Smoothing: {smoothed_fitness:.4f}")
        print(f"     • التحسن: {improvement:.4f}")
        
        # اختيار الأفضل
        if improvement > 0.001:
            final_solution = smoothed_solution
            final_fitness = smoothed_fitness
            print(f"   ✅ اعتماد Smoothing!")
        else:
            final_solution = solution
            final_fitness = fitness
            print(f"   ⚠ استخدام الحل الأصلي")
        
        print(f"\n   🏁 انتهى: اللياقة النهائية = {final_fitness:.4f}")
        
        return final_solution, final_fitness, time_pso_ga

# ============================================================
# مقيم آمن
# ============================================================
class SafeEvaluator:
    """مقيم آمن وبسيط"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, algorithm_name, algorithm_func):
        """تقييم آمن للخوارزمية"""
        print(f"\n{'='*60}")
        print(f"🧪 تقييم {algorithm_name}")
        print(f"{'='*60}")
        
        all_fitness = []
        all_lengths = []
        all_smoothness = []
        
        for run in range(SafeConfig.RUNS):
            print(f"\n   🔄 التشغيل {run+1}/{SafeConfig.RUNS}")
            
            solution, fitness, exec_time = algorithm_func(run + 1)
            
            # حساب المقاييس
            path = SafeGeometry.decode_path(solution)
            length = SafeGeometry.path_length(path)
            smoothness = SafeGeometry.smoothness_score(path)
            safety = SafeGeometry.safety_score(path)
            
            all_fitness.append(fitness)
            all_lengths.append(length)
            all_smoothness.append(smoothness)
            
            print(f"   ✓ النتائج:")
            print(f"     • اللياقة: {fitness:.4f}")
            print(f"     • الطول: {length:.2f} م")
            print(f"     • السلاسة: {smoothness:.3f}")
            print(f"     • الأمان: {safety:.3f}")
            print(f"     • الوقت: {exec_time:.2f} ث")
        
        # الإحصائيات
        avg_fitness = np.mean(all_fitness)
        avg_length = np.mean(all_lengths)
        avg_smoothness = np.mean(all_smoothness)
        
        # حساب النتيجة
        score = self.calculate_score(avg_fitness, avg_length, avg_smoothness)
        
        # حفظ النتائج
        self.results[algorithm_name] = {
            "avg_fitness": avg_fitness,
            "avg_length": avg_length,
            "avg_smoothness": avg_smoothness,
            "score": score,
            "all_fitness": all_fitness
        }
        
        # عرض النتائج
        self.display_results(algorithm_name, avg_fitness, avg_length, avg_smoothness, score)
    
    def calculate_score(self, fitness, length, smoothness):
        """حساب النتيجة"""
        # تطبيع
        norm_fitness = max(0, 1 - fitness * 3)
        norm_length = max(0, 1 - (length - SafeConfig.IDEAL_LENGTH) / 8.0)
        norm_smoothness = smoothness
        
        score = (0.4 * norm_fitness + 0.4 * norm_length + 0.2 * norm_smoothness) * 100
        return min(100.0, score)
    
    def display_results(self, name, fitness, length, smoothness, score):
        """عرض النتائج"""
        print(f"\n{'='*40}")
        print(f"📊 نتائج {name}")
        print(f"{'='*40}")
        print(f"🏆 النتيجة: {score:.1f}/100")
        print(f"📈 المتوسطات:")
        print(f"  • اللياقة: {fitness:.4f}")
        print(f"  • الطول: {length:.2f} م")
        print(f"  • السلاسة: {smoothness:.3f}")
    
    def compare(self):
        """مقارنة الخوارزميات"""
        if len(self.results) < 2:
            return
        
        print("\n" + "="*80)
        print("🏆 مقارنة الخوارزميات")
        print("="*80)
        
        print(f"\n{'الخوارزمية':<25} {'النتيجة':>8} {'اللياقة':>10} {'الطول':>10} {'السلاسة':>10}")
        print("-" * 73)
        
        for name, data in self.results.items():
            print(f"{name:<25} {data['score']:>8.1f} {data['avg_fitness']:>10.4f} "
                  f"{data['avg_length']:>10.2f} {data['avg_smoothness']:>10.3f}")
        
        print("-" * 73)
        
        # أفضل خوارزمية
        best = max(self.results.items(), key=lambda x: x[1]["score"])
        print(f"\n🎯 الأفضل: {best[0]} ({best[1]['score']:.1f}/100)")

# ============================================================
# الدالة الرئيسية
# ============================================================
def main():
    """الدالة الرئيسية الآمنة"""
    print("\n" + "="*70)
    print("🚀 بدء التشغيل الآمن - مقارنة خوارزميتين")
    print(f"📊 الإعدادات: {SafeConfig.RUNS} تشغيلات × {SafeConfig.ITERATIONS} تكرار")
    print("="*70)
    
    # إنشاء المقيم
    evaluator = SafeEvaluator()
    
    # دالة اللياقة
    fitness_func = SafeFitness.calculate
    
    # 1. PSO+GA
    hybrid_algo = SafeHybridPSOGA(fitness_func)
    
    # 2. PSO+GA+Smoothing
    smoothing_algo = SafePSOGASmoothing(fitness_func)
    
    # تشغيل وتقييم
    print("\n" + "="*70)
    print("1. PSO+GA الهجين")
    print("="*70)
    evaluator.evaluate("PSO+GA الهجين", hybrid_algo.run)
    
    print("\n" + "="*70)
    print("2. PSO+GA+Smoothing")
    print("="*70)
    evaluator.evaluate("PSO+GA+Smoothing", smoothing_algo.run)
    
    # المقارنة
    evaluator.compare()
    
    print("\n" + "="*70)
    print("✅ اكتمل بنجاح!")
    print("="*70)
    
    return evaluator

# ============================================================
# التشغيل
# ============================================================
if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ خطأ: {e}")
        import traceback
        traceback.print_exc()
