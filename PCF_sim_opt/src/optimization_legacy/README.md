# Legacy Optimization Code

이 폴더는 2025년 11월 재설계 이전의 최적화 코드를 보관하고 있습니다.

## 이동 날짜
- 2025-11-03

## 이동된 파일들

### 페이지 파일
- `page/optimization.py` → `page/optimization_legacy.py` (5,679줄)

### 코어 최적화 모듈
- `material_based_optimizer.py` (249,062 bytes)
  - 자재기반 최적화의 메인 옵티마이저
  - 복잡한 제약조건 및 시나리오 처리

- `material_reduction_manager.py` (24,715 bytes)
  - 자재별 감축 활동 관리
  - Formula 및 Ni/Co/Li 자재 분류

- `material_specific_objective.py` (8,307 bytes)
  - 자재별 목적함수 정의

- `material_constraints.py` (6,643 bytes)
  - 자재별 제약조건 관리

- `material_premium_cost.py` (18,145 bytes)
  - 프리미엄 비용 계산 (구버전, RE100PremiumCalculator로 대체됨)

## 새로운 시스템

### 위치
- `src/optimization_v2/` - 새로운 모듈식 최적화 시스템
- `page/optimization.py` - 새로운 최적화 페이지 (재작성)

### 개선사항
1. **모듈화**: 제약조건별로 독립적인 클래스 구조
2. **확장성**: 새로운 제약조건 타입을 쉽게 추가 가능
3. **유지보수성**: 작은 파일들로 분리 (~3,250줄 vs 5,679줄)
4. **사용자 친화성**: 직관적인 제약조건 설정 UI
5. **비용 통합**: RE100PremiumCalculator와 긴밀한 통합

## 참고용

이 코드는 참고 및 히스토리 목적으로 유지됩니다.
- 새 시스템에서 버그나 문제 발생 시 비교 참조
- 레거시 데이터 호환성 확인
- 알고리즘 검증

## 사용 중단 예정

새 시스템이 안정화되면 이 폴더는 완전히 삭제될 수 있습니다.
- 예상 삭제 시기: 2025년 12월 (새 시스템 검증 완료 후)

---

**중요**: 새로운 개발은 `src/optimization_v2/`에서 진행하세요!
