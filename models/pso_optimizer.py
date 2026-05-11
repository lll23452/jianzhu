"""PSO 粒子群优化 XGBoost 超参数"""
import numpy as np
from config import XGB_PARAM_BOUNDS


class PSOOptimizer:
    def __init__(self, n_particles=10, n_iter=30, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = XGB_PARAM_BOUNDS
        self.n_dims = len(self.bounds)

    @staticmethod
    def _decode_position(position):
        bounds = XGB_PARAM_BOUNDS
        return {
            'learning_rate': float(np.clip(position[0], bounds[0][0], bounds[0][1])),
            'max_depth': int(round(np.clip(position[1], bounds[1][0], bounds[1][1]))),
            'min_child_weight': int(round(np.clip(position[2], bounds[2][0], bounds[2][1]))),
            'subsample': float(np.clip(position[3], bounds[3][0], bounds[3][1])),
            'colsample_bytree': float(np.clip(position[4], bounds[4][0], bounds[4][1])),
            'n_estimators': int(round(np.clip(position[5], bounds[5][0], bounds[5][1]))),
            'reg_alpha': float(np.clip(position[6], bounds[6][0], bounds[6][1])),
            'reg_lambda': float(np.clip(position[7], bounds[7][0], bounds[7][1])),
        }

    def optimize(self, fitness_fn):
        bounds_arr = np.array(self.bounds)
        lb, ub = bounds_arr[:, 0], bounds_arr[:, 1]
        positions = np.random.uniform(lb, ub, (self.n_particles, self.n_dims))
        velocities = np.random.uniform(-1, 1, (self.n_particles, self.n_dims)) * (ub - lb) * 0.1
        pbest_positions = positions.copy()
        pbest_scores = np.array([fitness_fn(self._decode_position(p)) for p in positions])
        gbest_idx = np.argmin(pbest_scores)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        history = [gbest_score]
        for iteration in range(self.n_iter):
            r1, r2 = np.random.rand(2)
            velocities = (self.w * velocities
                          + self.c1 * r1 * (pbest_positions - positions)
                          + self.c2 * r2 * (gbest_position - positions))
            positions = np.clip(positions + velocities, lb, ub)
            for i in range(self.n_particles):
                score = fitness_fn(self._decode_position(positions[i]))
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()
                    if score < gbest_score:
                        gbest_score = score
                        gbest_position = positions[i].copy()
            history.append(gbest_score)
        return self._decode_position(gbest_position), gbest_score, history
