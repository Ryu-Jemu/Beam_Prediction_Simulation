"""
Position-Aware Beam Prediction Enhancement
위치 기반 빔 예측 정확도 향상 모듈

주요 기능:
1. 위치 기반 빔 탐색 범위 설정
2. 경계 반사 감지 및 처리
3. 물리적 제약 기반 빔 필터링
"""
import math
import numpy as np
import torch
from typing import Tuple, List, Optional


def compute_max_travel_distance(
    v_max: float,
    dt: float,
    steps: int
) -> float:
    """최대 이동 가능 거리 계산
    
    Args:
        v_max: 최대 속도 (m/s)
        dt: 시간 간격 (s)
        steps: 예측 스텝 수
    
    Returns:
        max_distance: 최대 이동 거리 (m)
    """
    return v_max * dt * steps


def compute_feasible_aod_range(
    current_pos: np.ndarray,
    bs_pos: np.ndarray,
    max_distance: float,
    num_samples: int = 360
) -> Tuple[np.ndarray, np.ndarray]:
    """현재 위치에서 도달 가능한 AoD 범위 계산
    
    Args:
        current_pos: [2] 현재 위치 (x, y)
        bs_pos: [2] 기지국 위치
        max_distance: 최대 이동 거리
        num_samples: 샘플링 포인트 수
    
    Returns:
        feasible_aods: [N] 가능한 AoD 배열
        feasible_positions: [N, 2] 해당 위치들
    """
    # 현재 위치를 중심으로 원형 샘플링
    angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
    
    feasible_positions = []
    feasible_aods = []
    
    for angle in angles:
        # 가능한 미래 위치
        future_x = current_pos[0] + max_distance * np.cos(angle)
        future_y = current_pos[1] + max_distance * np.sin(angle)
        future_pos = np.array([future_x, future_y])
        
        # BS에서 미래 위치로의 AoD
        dx = future_x - bs_pos[0]
        dy = future_y - bs_pos[1]
        aod = math.atan2(dy, dx)
        
        feasible_positions.append(future_pos)
        feasible_aods.append(aod)
    
    return np.array(feasible_aods), np.array(feasible_positions)


def compute_feasible_beam_range(
    current_pos: np.ndarray,
    bs_pos: np.ndarray,
    max_distance: float,
    num_beams: int = 64,
    margin: float = 0.1
) -> np.ndarray:
    """도달 가능한 빔 인덱스 범위 계산
    
    Args:
        current_pos: [2] 현재 위치
        bs_pos: [2] 기지국 위치
        max_distance: 최대 이동 거리
        num_beams: 빔 개수
        margin: 안전 마진 (0-1)
    
    Returns:
        feasible_beams: [K] 가능한 빔 인덱스 배열
    """
    # 가능한 AoD 범위 계산
    feasible_aods, _ = compute_feasible_aod_range(
        current_pos, bs_pos, max_distance * (1 + margin)
    )
    
    # AoD를 빔 인덱스로 변환
    aod_normalized = (feasible_aods + math.pi) / (2 * math.pi)
    beam_indices = (aod_normalized * (num_beams - 1)).astype(np.int32)
    beam_indices = np.clip(beam_indices, 0, num_beams - 1)
    
    # 유일한 빔 인덱스만 반환
    feasible_beams = np.unique(beam_indices)
    
    return feasible_beams


def filter_predictions_by_feasibility(
    predictions: torch.Tensor,
    current_pos: np.ndarray,
    bs_pos: np.ndarray,
    v_max: float,
    dt: float,
    num_beams: int = 64
) -> torch.Tensor:
    """물리적 제약을 고려하여 예측 필터링
    
    Args:
        predictions: [B, H, 2] 예측된 (sin, cos) AoD
        current_pos: [2] 현재 위치
        bs_pos: [2] 기지국 위치
        v_max: 최대 속도
        dt: 시간 간격
        num_beams: 빔 개수
    
    Returns:
        filtered_predictions: [B, H, 2] 필터링된 예측
    """
    B, H, _ = predictions.shape
    filtered = predictions.clone()
    
    # 각 예측 스텝마다 처리
    for h in range(H):
        max_distance = compute_max_travel_distance(v_max, dt, h + 1)
        
        # 가능한 빔 범위
        feasible_beams = compute_feasible_beam_range(
            current_pos, bs_pos, max_distance, num_beams
        )
        
        # 예측된 AoD
        pred_h = predictions[:, h, :]  # [B, 2]
        pred_angles = torch.atan2(pred_h[:, 0], pred_h[:, 1])  # [B]
        
        # 빔 인덱스로 변환
        pred_normalized = (pred_angles + math.pi) / (2 * math.pi)
        pred_beams = (pred_normalized * (num_beams - 1)).long()
        pred_beams = torch.clamp(pred_beams, 0, num_beams - 1)
        
        # 불가능한 빔 감지
        feasible_set = set(feasible_beams)
        
        for b in range(B):
            beam_idx = pred_beams[b].item()
            if beam_idx not in feasible_set:
                # 가장 가까운 가능한 빔으로 교체
                closest_beam = min(feasible_beams, 
                                   key=lambda x: abs(x - beam_idx))
                
                # 빔을 AoD로 변환
                corrected_aod = (closest_beam / (num_beams - 1)) * 2 * math.pi - math.pi
                
                # (sin, cos)로 변환
                filtered[b, h, 0] = math.sin(corrected_aod)
                filtered[b, h, 1] = math.cos(corrected_aod)
    
    return filtered


# ==================== 경계 반사 처리 ====================

def detect_boundary_proximity(
    pos: np.ndarray,
    area_size: float,
    margin: float = 10.0
) -> Tuple[bool, str]:
    """경계 근접 감지
    
    Args:
        pos: [2] 위치 (x, y)
        area_size: 영역 크기
        margin: 경계 마진 (m)
    
    Returns:
        is_near: 경계 근처 여부
        boundary: 'left', 'right', 'top', 'bottom', 'corner', 'none'
    """
    x, y = pos[0], pos[1]
    
    near_left = x < margin
    near_right = x > area_size - margin
    near_bottom = y < margin
    near_top = y > area_size - margin
    
    # 코너
    if (near_left or near_right) and (near_top or near_bottom):
        return True, 'corner'
    
    # 경계
    if near_left:
        return True, 'left'
    if near_right:
        return True, 'right'
    if near_bottom:
        return True, 'bottom'
    if near_top:
        return True, 'top'
    
    return False, 'none'


def compute_reflected_heading(
    heading: float,
    pos: np.ndarray,
    area_size: float
) -> float:
    """경계 반사 후 방향각 계산
    
    Args:
        heading: 현재 방향각 (rad)
        pos: 현재 위치
        area_size: 영역 크기
    
    Returns:
        reflected_heading: 반사 후 방향각
    """
    is_near, boundary = detect_boundary_proximity(pos, area_size, margin=10.0)
    
    if not is_near:
        return heading
    
    # 반사 계산
    if boundary in ['left', 'right']:
        # X 경계: heading → π - heading
        reflected = math.pi - heading
    elif boundary in ['top', 'bottom']:
        # Y 경계: heading → -heading
        reflected = -heading
    elif boundary == 'corner':
        # 코너: 180도 반전
        reflected = heading + math.pi
    else:
        reflected = heading
    
    # [-π, π]로 정규화
    reflected = (reflected + math.pi) % (2 * math.pi) - math.pi
    
    return reflected


def enhance_prediction_near_boundary(
    predictions: torch.Tensor,
    trajectory: np.ndarray,
    bs_pos: np.ndarray,
    area_size: float,
    margin: float = 15.0
) -> torch.Tensor:
    """경계 근처에서 예측 향상
    
    Args:
        predictions: [B, H, 2] 예측
        trajectory: [B, U+H, 2] 전체 궤적
        bs_pos: [2] 기지국 위치
        area_size: 영역 크기
        margin: 경계 마진
    
    Returns:
        enhanced_predictions: [B, H, 2] 향상된 예측
    """
    B, H, _ = predictions.shape
    enhanced = predictions.clone()
    
    for b in range(B):
        # 과거 궤적에서 마지막 위치와 방향 추출
        traj_b = trajectory[b].cpu().numpy()  # [U+H, 2]
        
        # 과거 마지막 위치 (U-1 인덱스)
        # 주의: trajectory는 전체 U+H, predictions는 미래 H만
        # 여기서는 간단히 마지막 알려진 위치 사용
        # 실제로는 U 위치를 정확히 추적해야 함
        
        for h in range(H):
            # 현재 스텝의 예상 위치 (근사)
            # 실제 구현에서는 CTRV 모델로 위치 예측
            
            # 예측된 AoD
            pred_aod = torch.atan2(enhanced[b, h, 0], enhanced[b, h, 1]).item()
            
            # 경계 근처 체크 (간단한 버전)
            # 실제로는 예측된 위치를 사용해야 함
            # 여기서는 AoD만으로 판단
            
            # 경계 근처에서 빔 변화가 급격할 가능성 고려
            # 실제 개선은 더 정교한 모델링 필요
            
            pass  # 실제 구현에서는 여기에 로직 추가
    
    return enhanced


# ==================== 통합 인터페이스 ====================

class PositionAwareBeamPredictor:
    """위치 인식 빔 예측기
    
    기존 모델에 위치 기반 제약과 경계 처리를 추가
    """
    
    def __init__(
        self,
        model,
        cfg,
        bs_pos: np.ndarray
    ):
        """
        Args:
            model: 기존 빔 예측 모델
            cfg: 설정
            bs_pos: 기지국 위치
        """
        self.model = model
        self.cfg = cfg
        self.bs_pos = bs_pos
        
        # 최대 이동 거리 계산
        self.max_distance = compute_max_travel_distance(
            cfg.speed_max_mps,
            cfg.delta_t_s,
            cfg.H
        )
    
    def predict(
        self,
        x: torch.Tensor,
        current_pos: np.ndarray,
        stats_text: Optional[str] = None,
        apply_constraints: bool = True
    ) -> torch.Tensor:
        """위치 제약을 고려한 예측
        
        Args:
            x: [B, C, U] 입력 특징
            current_pos: [2] 현재 위치
            stats_text: 통계 텍스트
            apply_constraints: 제약 적용 여부
        
        Returns:
            predictions: [B, H, 2] 예측 (sin, cos)
        """
        # 기본 모델 예측
        predictions = self.model(x, stats_text)
        
        if not apply_constraints:
            return predictions
        
        # 물리적 제약 적용
        predictions = filter_predictions_by_feasibility(
            predictions,
            current_pos,
            self.bs_pos,
            self.cfg.speed_max_mps,
            self.cfg.delta_t_s,
            self.cfg.M
        )
        
        return predictions
    
    def get_feasible_beams(
        self,
        current_pos: np.ndarray,
        horizon: Optional[int] = None
    ) -> np.ndarray:
        """현재 위치에서 가능한 빔 목록
        
        Args:
            current_pos: 현재 위치
            horizon: 예측 horizon (None이면 cfg.H 사용)
        
        Returns:
            feasible_beams: 가능한 빔 인덱스 배열
        """
        if horizon is None:
            horizon = self.cfg.H
        
        max_dist = compute_max_travel_distance(
            self.cfg.speed_max_mps,
            self.cfg.delta_t_s,
            horizon
        )
        
        return compute_feasible_beam_range(
            current_pos,
            self.bs_pos,
            max_dist,
            self.cfg.M
        )


# ==================== 유틸리티 함수 ====================

def compute_trajectory_statistics(
    trajectory: np.ndarray,
    area_size: float
) -> dict:
    """궤적 통계 계산
    
    Args:
        trajectory: [T, 2] 궤적
        area_size: 영역 크기
    
    Returns:
        stats: 통계 딕셔너리
    """
    # 총 이동 거리
    diffs = np.diff(trajectory, axis=0)
    distances = np.sqrt((diffs**2).sum(axis=1))
    total_distance = distances.sum()
    
    # 직선 거리
    straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
    
    # 경로 효율성
    efficiency = straight_distance / (total_distance + 1e-8)
    
    # 경계 반사 횟수
    reflections = 0
    for t in range(len(trajectory) - 1):
        pos_t = trajectory[t]
        pos_t1 = trajectory[t + 1]
        
        # 경계 충돌 체크
        if (pos_t[0] <= 0 and pos_t1[0] > 0) or \
           (pos_t[0] >= area_size and pos_t1[0] < area_size) or \
           (pos_t[1] <= 0 and pos_t1[1] > 0) or \
           (pos_t[1] >= area_size and pos_t1[1] < area_size):
            reflections += 1
    
    # 영역 커버리지
    x_range = trajectory[:, 0].max() - trajectory[:, 0].min()
    y_range = trajectory[:, 1].max() - trajectory[:, 1].min()
    coverage = (x_range * y_range) / (area_size ** 2)
    
    return {
        'total_distance': float(total_distance),
        'straight_distance': float(straight_distance),
        'efficiency': float(efficiency),
        'reflections': int(reflections),
        'coverage': float(coverage),
        'x_range': float(x_range),
        'y_range': float(y_range)
    }


if __name__ == "__main__":
    # 테스트
    print("Position-Aware Beam Prediction Enhancement")
    print("="*50)
    
    # 테스트 파라미터
    current_pos = np.array([50.0, 50.0])
    bs_pos = np.array([100.0, 100.0])
    v_max = 15.0
    dt = 0.1
    H = 20
    
    # 최대 이동 거리
    max_dist = compute_max_travel_distance(v_max, dt, H)
    print(f"Max travel distance: {max_dist:.1f} m")
    
    # 가능한 AoD 범위
    feasible_aods, _ = compute_feasible_aod_range(
        current_pos, bs_pos, max_dist
    )
    print(f"Feasible AoD range: [{feasible_aods.min():.2f}, {feasible_aods.max():.2f}] rad")
    
    # 가능한 빔 범위
    feasible_beams = compute_feasible_beam_range(
        current_pos, bs_pos, max_dist, num_beams=64
    )
    print(f"Feasible beams: {len(feasible_beams)}/{64} beams")
    print(f"Beam indices: {feasible_beams[:10]}... (showing first 10)")
    
    # 경계 근접 테스트
    test_positions = [
        np.array([5.0, 50.0]),    # 좌측 경계
        np.array([195.0, 50.0]),  # 우측 경계
        np.array([50.0, 5.0]),    # 하단 경계
        np.array([5.0, 5.0]),     # 코너
        np.array([100.0, 100.0])  # 중앙
    ]
    
    print("\nBoundary proximity test:")
    for pos in test_positions:
        is_near, boundary = detect_boundary_proximity(pos, 200.0)
        print(f"  Pos {pos}: near={is_near}, boundary={boundary}")
