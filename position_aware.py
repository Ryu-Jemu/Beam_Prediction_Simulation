"""
Advanced Beam Predictor with Physics-based Logic
물리 기반 로직과 강건한 예측을 위한 고급 빔 예측기

주요 기능:
1. Heading 추정 및 예측 범위 계산
2. 기지국 통과 감지 및 특별 처리
3. 벽 반사 예측
4. Kalman Filter 기반 추적
5. Adaptive Beam Sweeping
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
import math
from dataclasses import dataclass
from scipy.stats import norm
from filterpy.kalman import KalmanFilter


@dataclass
class PredictionContext:
    """예측 컨텍스트 정보"""
    user_pos: np.ndarray  # [2] 현재 위치
    past_positions: np.ndarray  # [N, 2] 과거 위치들
    past_beams: np.ndarray  # [N] 과거 빔 인덱스
    bs_pos: np.ndarray  # [2] 기지국 위치
    area_size: float  # 영역 크기
    speed: float  # 현재 속도
    heading: float  # 현재 방향
    uncertainty: float  # 불확실성 (0-1)


class PhysicsBasedBeamPredictor:
    """물리 기반 빔 예측기"""

    def __init__(
        self,
        num_beams: int = 64,
        area_size: float = 200.0,
        bs_pos: Optional[np.ndarray] = None,
        dt: float = 0.1
    ):
        """
        Args:
            num_beams: 빔 개수
            area_size: 영역 크기 (m)
            bs_pos: 기지국 위치
            dt: 시간 간격 (s)
        """
        self.num_beams = num_beams
        self.area_size = area_size
        self.bs_pos = bs_pos if bs_pos is not None else np.array([area_size/2, area_size/2])
        self.dt = dt

        # Kalman Filter 초기화
        self.kf = self._init_kalman_filter()

        # 통계 수집
        self.prediction_history = []
        self.error_history = []

    def _init_kalman_filter(self) -> KalmanFilter:
        """Kalman Filter 초기화 (위치, 속도 추적)"""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]

        # State transition matrix
        kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise
        kf.Q = np.eye(4) * 0.1
        kf.Q[2:, 2:] *= 0.5  # 속도 노이즈

        # Measurement noise
        kf.R = np.eye(2) * 1.0

        # Initial uncertainty
        kf.P = np.eye(4) * 100

        return kf

    def estimate_heading(
        self,
        positions: np.ndarray,
        window: int = 3
    ) -> Tuple[float, float]:
        """과거 위치로부터 heading 추정

        Args:
            positions: [N, 2] 과거 위치들
            window: 사용할 위치 개수

        Returns:
            heading: 방향 (radians)
            confidence: 신뢰도 (0-1)
        """
        if len(positions) < 2:
            return 0.0, 0.0

        # 최근 window 개 위치 사용
        recent = positions[-min(window, len(positions)):]

        # 속도 벡터 계산
        velocities = np.diff(recent, axis=0)

        if len(velocities) == 0:
            return 0.0, 0.0

        # 가중 평균 (최근일수록 높은 가중치)
        weights = np.exp(np.linspace(-1, 0, len(velocities)))
        weights /= weights.sum()

        avg_velocity = np.average(velocities, axis=0, weights=weights)

        # Heading 계산
        heading = np.arctan2(avg_velocity[1], avg_velocity[0])

        # 신뢰도 계산 (속도 벡터들의 일관성)
        if len(velocities) > 1:
            angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            std_angle = np.std(angles)
            confidence = np.exp(-std_angle)  # 표준편차가 작을수록 높은 신뢰도
        else:
            confidence = 0.5

        return heading, confidence

    def predict_beam_range(
        self,
        context: PredictionContext,
        horizon: int = 1
    ) -> Tuple[int, int, float]:
        """예측 빔 범위 계산

        Args:
            context: 예측 컨텍스트
            horizon: 예측 시점 (steps ahead)

        Returns:
            min_beam: 최소 빔 인덱스
            max_beam: 최대 빔 인덱스
            center_beam: 중심 빔 (float)
        """
        # 1. Kalman Filter로 미래 위치 예측
        future_pos = self._predict_future_position(context, horizon)

        # 2. 예측 위치에서의 AoD 계산
        dx = future_pos[0] - self.bs_pos[0]
        dy = future_pos[1] - self.bs_pos[1]
        predicted_aod = np.arctan2(dy, dx)

        # 3. AoD를 빔 인덱스로 변환
        center_beam = self._aod_to_beam_index(predicted_aod, continuous=True)

        # 4. 불확실성 기반 범위 계산
        uncertainty_beams = max(1, int(self.num_beams * context.uncertainty * 0.2))

        min_beam = int(max(0, center_beam - uncertainty_beams))
        max_beam = int(min(self.num_beams - 1, center_beam + uncertainty_beams))

        return min_beam, max_beam, center_beam

    def _predict_future_position(
        self,
        context: PredictionContext,
        horizon: int
    ) -> np.ndarray:
        """Kalman Filter로 미래 위치 예측"""
        # 현재 상태 업데이트
        self.kf.predict()
        self.kf.update(context.user_pos)

        # 미래 예측
        future_state = self.kf.x.copy()
        F_horizon = np.linalg.matrix_power(self.kf.F, horizon)
        future_state = F_horizon @ future_state

        return future_state[:2]

    def handle_bs_crossing(
        self,
        context: PredictionContext
    ) -> Optional[int]:
        """기지국 통과 감지 및 처리

        Returns:
            predicted_beam: 예측 빔 (기지국 통과 시)
        """
        # 기지국까지 거리
        dist_to_bs = np.linalg.norm(context.user_pos - self.bs_pos)

        # 접근 속도 (radial velocity)
        if len(context.past_positions) >= 2:
            prev_dist = np.linalg.norm(context.past_positions[-2] - self.bs_pos)
            radial_velocity = (prev_dist - dist_to_bs) / self.dt
        else:
            radial_velocity = 0

        # 통과 임박 감지 (거리 < 20m and 접근 중)
        if dist_to_bs < 20.0 and radial_velocity > 0:
            # 통과 후 예상 방향 계산
            future_heading = context.heading

            # 반대편 위치 예측
            future_pos = self.bs_pos + np.array([
                np.cos(future_heading) * 20,
                np.sin(future_heading) * 20
            ])

            # 빔 인덱스 계산
            dx = future_pos[0] - self.bs_pos[0]
            dy = future_pos[1] - self.bs_pos[1]
            future_aod = np.arctan2(dy, dx)

            return self._aod_to_beam_index(future_aod)

        return None

    def predict_reflection(
        self,
        context: PredictionContext
    ) -> List[int]:
        """벽 반사 예측

        Returns:
            possible_beams: 가능한 빔 인덱스 리스트
        """
        possible_beams = []
        margin = 20.0  # 벽 근접 마진

        x, y = context.user_pos
        heading = context.heading

        # 각 벽에 대해 반사 검사
        reflections = []

        # 좌측 벽
        if x < margin and np.cos(heading) < 0:
            reflected_heading = np.pi - heading
            reflections.append(reflected_heading)

        # 우측 벽
        if x > self.area_size - margin and np.cos(heading) > 0:
            reflected_heading = np.pi - heading
            reflections.append(reflected_heading)

        # 하단 벽
        if y < margin and np.sin(heading) < 0:
            reflected_heading = -heading
            reflections.append(reflected_heading)

        # 상단 벽
        if y > self.area_size - margin and np.sin(heading) > 0:
            reflected_heading = -heading
            reflections.append(reflected_heading)

        # 반사 방향에 해당하는 빔 계산
        for ref_heading in reflections:
            # 반사 후 위치 예측
            future_pos = context.user_pos + np.array([
                np.cos(ref_heading) * context.speed * self.dt * 5,  # 5 steps ahead
                np.sin(ref_heading) * context.speed * self.dt * 5
            ])

            # 빔 인덱스 계산
            dx = future_pos[0] - self.bs_pos[0]
            dy = future_pos[1] - self.bs_pos[1]
            aod = np.arctan2(dy, dx)
            beam_idx = self._aod_to_beam_index(aod)

            if 0 <= beam_idx < self.num_beams:
                possible_beams.append(beam_idx)

        return possible_beams

    def adaptive_beam_sweep(
        self,
        context: PredictionContext,
        base_prediction: int
    ) -> List[Tuple[int, float]]:
        """적응형 빔 스위핑

        Args:
            context: 예측 컨텍스트
            base_prediction: 기본 예측 빔

        Returns:
            beam_candidates: [(beam_idx, probability)] 리스트
        """
        candidates = []

        # 불확실성에 따른 스위핑 범위
        sweep_range = max(1, int(self.num_beams * context.uncertainty * 0.1))

        # Gaussian 분포로 확률 할당
        for offset in range(-sweep_range, sweep_range + 1):
            beam_idx = base_prediction + offset
            if 0 <= beam_idx < self.num_beams:
                # 중심에서 멀수록 낮은 확률
                prob = norm.pdf(offset, 0, sweep_range/2)
                candidates.append((beam_idx, prob))

        # 정규화
        total_prob = sum(p for _, p in candidates)
        candidates = [(b, p/total_prob) for b, p in candidates]

        return candidates

    def predict(
        self,
        past_beams: np.ndarray,
        past_positions: np.ndarray,
        current_pos: np.ndarray,
        horizon: int = 1
    ) -> Tuple[int, Dict[str, float]]:
        """통합 빔 예측

        Args:
            past_beams: [N] 과거 빔 인덱스
            past_positions: [N, 2] 과거 위치들
            current_pos: [2] 현재 위치
            horizon: 예측 시점

        Returns:
            predicted_beam: 예측 빔 인덱스
            metrics: 예측 메트릭
        """
        # 컨텍스트 생성
        heading, confidence = self.estimate_heading(
            np.vstack([past_positions, current_pos])
        )

        speed = np.linalg.norm(current_pos - past_positions[-1]) / self.dt if len(past_positions) > 0 else 10.0

        context = PredictionContext(
            user_pos=current_pos,
            past_positions=past_positions,
            past_beams=past_beams,
            bs_pos=self.bs_pos,
            area_size=self.area_size,
            speed=speed,
            heading=heading,
            uncertainty=1.0 - confidence
        )

        # 1. 기지국 통과 체크
        bs_crossing_beam = self.handle_bs_crossing(context)
        if bs_crossing_beam is not None:
            return bs_crossing_beam, {"method": "bs_crossing", "confidence": 0.9}

        # 2. 벽 반사 예측
        reflection_beams = self.predict_reflection(context)
        if reflection_beams:
            # 가장 가능성 높은 반사 빔 선택
            return reflection_beams[0], {"method": "reflection", "confidence": 0.8}

        # 3. 일반 예측
        min_beam, max_beam, center_beam = self.predict_beam_range(context, horizon)

        # 4. 적응형 스위핑
        candidates = self.adaptive_beam_sweep(context, int(center_beam))

        # 최고 확률 빔 선택
        best_beam = max(candidates, key=lambda x: x[1])[0]

        return best_beam, {
            "method": "adaptive",
            "confidence": confidence,
            "uncertainty": context.uncertainty,
            "center_beam": center_beam
        }

    def _aod_to_beam_index(self, aod: float, continuous: bool = False) -> float:
        """AoD를 빔 인덱스로 변환"""
        normalized = (aod + np.pi) / (2 * np.pi)
        beam = normalized * (self.num_beams - 1)

        if continuous:
            return beam
        else:
            return int(np.clip(np.round(beam), 0, self.num_beams - 1))

    def _beam_index_to_aod(self, beam_idx: int) -> float:
        """빔 인덱스를 AoD로 변환"""
        normalized = beam_idx / (self.num_beams - 1)
        return normalized * 2 * np.pi - np.pi

    def compute_error(
        self,
        predicted_beam: int,
        actual_beam: int
    ) -> float:
        """예측 오차 계산 (각도 차이)"""
        pred_aod = self._beam_index_to_aod(predicted_beam)
        actual_aod = self._beam_index_to_aod(actual_beam)

        # 각도 차이 (circular)
        diff = np.abs(pred_aod - actual_aod)
        diff = min(diff, 2*np.pi - diff)

        return np.degrees(diff)  # degrees로 반환


class HybridBeamPredictor(nn.Module):
    """물리 기반 + 학습 기반 하이브리드 예측기"""

    def __init__(
        self,
        physics_predictor: PhysicsBasedBeamPredictor,
        learning_model: Optional[nn.Module] = None,
        alpha: float = 0.5
    ):
        """
        Args:
            physics_predictor: 물리 기반 예측기
            learning_model: 학습 기반 모델 (optional)
            alpha: 물리 기반 가중치 (0-1)
        """
        super().__init__()
        self.physics_predictor = physics_predictor
        self.learning_model = learning_model
        self.alpha = alpha

    def forward(
        self,
        x: torch.Tensor,
        past_positions: np.ndarray,
        current_pos: np.ndarray
    ) -> Tuple[torch.Tensor, Dict]:
        """하이브리드 예측

        Args:
            x: [B, C, U] 입력 특징
            past_positions: 과거 위치
            current_pos: 현재 위치

        Returns:
            predictions: [B, H, 2] (sin, cos)
            info: 추가 정보
        """
        B, _, _ = x.shape
        H = 5  # 예측 horizon

        predictions = torch.zeros(B, H, 2)
        info = {"physics_weight": self.alpha}

        for b in range(B):
            # 물리 기반 예측
            for h in range(H):
                # 과거 빔 추출 (간단화)
                past_beams = x[b, 0, -3:].cpu().numpy()  # 최근 3개
                past_beams = ((past_beams + 1) / 2 * (self.physics_predictor.num_beams - 1)).astype(int)

                physics_beam, metrics = self.physics_predictor.predict(
                    past_beams,
                    past_positions,
                    current_pos,
                    horizon=h+1
                )

                # 빔을 (sin, cos)로 변환
                aod = self.physics_predictor._beam_index_to_aod(physics_beam)
                predictions[b, h, 0] = np.sin(aod)
                predictions[b, h, 1] = np.cos(aod)

        # 학습 기반 예측과 결합 (있는 경우)
        if self.learning_model is not None:
            with torch.no_grad():
                learning_pred = self.learning_model(x)
            predictions = self.alpha * predictions + (1 - self.alpha) * learning_pred

        return predictions, info


def evaluate_predictor(
    predictor: PhysicsBasedBeamPredictor,
    test_data: List[Dict],
    verbose: bool = True
) -> Dict[str, float]:
    """예측기 평가

    Args:
        predictor: 빔 예측기
        test_data: 테스트 데이터
        verbose: 상세 출력

    Returns:
        metrics: 평가 메트릭
    """
    total_error = 0
    errors_by_method = {"adaptive": [], "bs_crossing": [], "reflection": []}

    for sample in test_data:
        past_beams = sample["past_beams"]
        past_positions = sample["past_positions"]
        current_pos = sample["current_pos"]
        actual_beam = sample["actual_beam"]

        predicted_beam, info = predictor.predict(
            past_beams,
            past_positions,
            current_pos
        )

        error = predictor.compute_error(predicted_beam, actual_beam)
        total_error += error

        method = info["method"]
        if method in errors_by_method:
            errors_by_method[method].append(error)

    # 통계 계산
    mae = total_error / len(test_data)

    results = {
        "mae_degrees": mae,
        "total_samples": len(test_data)
    }

    # 방법별 통계
    for method, errors in errors_by_method.items():
        if errors:
            results[f"mae_{method}"] = np.mean(errors)
            results[f"count_{method}"] = len(errors)

    if verbose:
        print("=" * 60)
        print("BEAM PREDICTION EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall MAE: {mae:.2f}°")
        print(f"Total Samples: {len(test_data)}")
        print("\nPer-Method Performance:")
        for method in ["adaptive", "bs_crossing", "reflection"]:
            if f"mae_{method}" in results:
                print(f"  {method:12s}: MAE={results[f'mae_{method}']:.2f}° (n={results[f'count_{method}']})")
        print("=" * 60)

    return results


if __name__ == "__main__":
    print("Advanced Beam Predictor Test")
    print("=" * 60)

    # 예측기 생성
    predictor = PhysicsBasedBeamPredictor(
        num_beams=64,
        area_size=200.0,
        dt=0.1
    )

    # 테스트 데이터 생성
    # NOTE: a fixed seed (e.g., 42) is not strictly necessary here.  We
    # avoid seeding the global RNG so that each run produces different
    # random test trajectories.  If reproducibility is needed, set
    # `np.random.seed(...)` externally.
    test_data = []

    for _ in range(100):
        # 랜덤 궤적 생성
        trajectory = np.random.rand(10, 2) * 200
        past_positions = trajectory[:7]
        current_pos = trajectory[7]

        # 빔 인덱스 계산
        past_beams = []
        for pos in past_positions:
            dx = pos[0] - predictor.bs_pos[0]
            dy = pos[1] - predictor.bs_pos[1]
            aod = np.arctan2(dy, dx)
            beam = predictor._aod_to_beam_index(aod)
            past_beams.append(beam)

        # 실제 빔
        dx = trajectory[8, 0] - predictor.bs_pos[0]
        dy = trajectory[8, 1] - predictor.bs_pos[1]
        actual_aod = np.arctan2(dy, dx)
        actual_beam = predictor._aod_to_beam_index(actual_aod)

        test_data.append({
            "past_beams": np.array(past_beams[-3:]),
            "past_positions": past_positions,
            "current_pos": current_pos,
            "actual_beam": actual_beam
        })

    # 평가
    results = evaluate_predictor(predictor, test_data)

    print(f"\n✓ Test completed successfully!")
    print(f"  MAE is per-step angle difference (not cumulative)")