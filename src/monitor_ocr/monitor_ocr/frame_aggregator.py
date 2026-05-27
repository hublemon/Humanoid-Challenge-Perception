"""
FrameAggregator: 여러 프레임 OCR 결과를 합쳐 더 신뢰할 수 있는 값 추출

정적 데이터 (프레임 간 변하지 않음): 포인트, 제목, 총점
  → 슬라이딩 윈도우에서 다수결/최빈값

동적 데이터 (실시간으로 변함): 진행률%, 버튼 상태
  → 최신 프레임 사용 (다수결 부적합)
"""
from collections import Counter


class FrameAggregator:
    def __init__(self, window: int = 10, btn_window: int = 3):
        """
        window     : 정적 데이터 집계에 사용할 최근 프레임 수
        btn_window : 버튼 다수결에 사용할 최근 프레임 수
        """
        self.window     = window
        self.btn_window = btn_window
        self._history   = []          # 최근 window개 결과 보관

        # 한 번 확정되면 고정되는 값
        self._locked_total  = None    # 총점 (0에서 시작, 게임 중 변함)
        self._locked_points = [None, None, None]  # 포인트는 변하지 않음

    # ── 업데이트 ─────────────────────────────────────────────────────────────

    def update(self, result: dict) -> dict:
        """새 프레임 결과를 추가하고 집계된 최종 결과 반환."""
        self._history.append(result)
        if len(self._history) > self.window:
            self._history.pop(0)
        return self._aggregate()

    # ── 집계 ─────────────────────────────────────────────────────────────────

    def _aggregate(self) -> dict:
        hist = self._history

        # 1. 포인트: 다수결 (한 번 확정되면 lock)
        points = []
        for i in range(3):
            if self._locked_points[i] is not None:
                points.append(self._locked_points[i])
                continue
            vals = [h["mission_points"][i] for h in hist
                    if h.get("mission_points") and h["mission_points"][i] is not None]
            if vals:
                winner, cnt = Counter(vals).most_common(1)[0]
                # 윈도우의 60% 이상이 같은 값 → 확정
                if cnt >= max(2, len(hist) * 0.6):
                    self._locked_points[i] = winner
                points.append(winner)
            else:
                points.append(None)

        # 2. 버튼: 최근 btn_window 프레임 다수결 (btn_active 또는 btn_text 둘 다 지원)
        def _is_btn(h):
            if "btn_active" in h:
                return bool(h["btn_active"])
            # 구버전 JSON 호환: btn_text가 있으면 활성
            return bool(h.get("btn_text", ""))
        recent_btn = [_is_btn(h) for h in hist[-self.btn_window:]]
        btn_active = (recent_btn.count(True) > len(recent_btn) / 2) if recent_btn else False

        # 3. 진행률: 최신 프레임 값 (동적, 실시간 추적)
        latest = hist[-1]
        mission_percents = latest.get("mission_percents", [None, None, None])

        # 4. 제목: 토큰 빈도 → 가장 많은 토큰 순으로 재조합
        title = self._best_title()

        # 5. 총점: 유효한 값이 나오면 lock (0P에서 시작, 증가만 함)
        totals = [h["total_points"] for h in hist
                  if h.get("total_points") is not None]
        if totals:
            # 가장 큰 값 우선 (점수는 증가만 하므로)
            candidate = max(totals)
            if self._locked_total is None or candidate > self._locked_total:
                self._locked_total = candidate
        total_points = self._locked_total

        # 6. 미션 텍스트: 최빈값 (오인식이 많아 참고용)
        mission_texts = self._best_texts()

        return {
            "frames_used":      len(hist),
            "mission_points":   points,
            "btn_active":       btn_active,
            "mission_percents": mission_percents,
            "title":            title,
            "total_points":     total_points,
            "mission_texts":    mission_texts,
            # 최신 프레임의 원본 정보
            "latest_elapsed_ms":    latest.get("elapsed_ms"),
            "latest_screen_detected": latest.get("screen_detected"),
        }

    @staticmethod
    def _is_korean_token(tok: str) -> bool:
        """한글 음절이 50% 이상인 토큰만 유효."""
        if len(tok) < 2:
            return False
        korean = sum(1 for c in tok if '가' <= c <= '힣')
        return korean / len(tok) >= 0.5

    def _best_title(self) -> str:
        """한국어 토큰 빈도 기반으로 제목 재조합."""
        all_tokens = []
        for h in self._history:
            t = h.get("title", "")
            if t:
                # 한국어 토큰만 수집 (잡음 "B&", "G&" 등 제거)
                all_tokens.extend(tok for tok in t.split() if self._is_korean_token(tok))
        if not all_tokens:
            return ""
        freq = Counter(all_tokens)
        top = {tok for tok, _ in freq.most_common(6)}

        # 어순 추정: top 토큰의 평균 출현 위치
        pos = {}
        for h in self._history:
            tokens = [tok for tok in h.get("title", "").split() if self._is_korean_token(tok)]
            for i, tok in enumerate(tokens):
                if tok in top:
                    pos.setdefault(tok, []).append(i)
        ordered = sorted(
            [(tok, sum(ps) / len(ps)) for tok, ps in pos.items()],
            key=lambda x: x[1]
        )
        return " ".join(tok for tok, _ in ordered[:4])

    def _best_texts(self) -> list:
        """각 미션 행별 최빈 텍스트 (신뢰도 낮음, 참고용)."""
        result = []
        for i in range(3):
            texts = [h["mission_texts"][i] for h in self._history
                     if h.get("mission_texts") and h["mission_texts"][i]]
            if texts:
                result.append(Counter(texts).most_common(1)[0][0])
            else:
                result.append("")
        return result

    def reset(self):
        """상태 초기화 (새 미션 라운드 시작 시)."""
        self._history.clear()
        self._locked_total  = None
        self._locked_points = [None, None, None]


class FrameAggregatorParts:
    """
    부품 수량 테이블 OCR 결과 집계 (슬라이딩 윈도우 다수결).

    수량은 프레임 간 변하지 않으므로 다수결로 안정화.
    이름은 행 순서 기반 PART_NAMES 고정값 사용.
    """

    def __init__(self, window: int = 10):
        self.window   = window
        self._history = []

    def update(self, result: dict) -> dict:
        # 화면은 감지됐더라도 OCR로 읽힌 수량이 하나도 없으면 미감지로 처리
        if result.get("screen_detected") and result.get("parts"):
            if all(p["count"] < 0 for p in result["parts"]):
                result = dict(result, screen_detected=False)

        self._history.append(result)
        if len(self._history) > self.window:
            self._history.pop(0)
        return self._aggregate()

    def _aggregate(self) -> dict:
        from monitor_ocr.ocr_pipeline_parts import PART_NAMES, N_ROWS

        hist   = self._history
        latest = hist[-1]

        # 최신 프레임에서 화면 미감지 → 즉시 -1 반환 (이전 값 유지 안 함)
        if not latest.get("screen_detected", False):
            return {
                "frames_used":            len(hist),
                "parts":                  [{"name": PART_NAMES[i], "count": -1} for i in range(N_ROWS)],
                "latest_elapsed_ms":      latest.get("elapsed_ms"),
                "latest_screen_detected": False,
            }

        # 화면 감지된 경우: 유효값의 다수결
        parts = []
        for i in range(N_ROWS):
            counts = [
                h["parts"][i]["count"]
                for h in hist
                if h.get("screen_detected") and h.get("parts")
                and len(h["parts"]) > i
                and h["parts"][i]["count"] >= 0
            ]
            count = Counter(counts).most_common(1)[0][0] if counts else -1
            parts.append({"name": PART_NAMES[i], "count": count})

        return {
            "frames_used":            len(hist),
            "parts":                  parts,
            "latest_elapsed_ms":      latest.get("elapsed_ms"),
            "latest_screen_detected": True,
        }

    def reset(self):
        self._history.clear()
