"""
Type A 양극재 구성에 대한 제약조건 적용 수정 모듈

이 모듈은 Type A 양극재 구성에서 recycle_ratio와 low_carbon_ratio가
상수값으로 설정되어 있을 때 제약조건 적용 방식을 수정합니다.
"""

from pyomo.environ import ConcreteModel, Constraint


def add_type_a_material_constraints(model: ConcreteModel) -> None:
    """
    Type A 양극재 설정에서 자재 제약조건 추가
    
    Args:
        model: Pyomo 모델
    """
    # 이미 제약조건이 있는지 확인하고, 있으면 반환
    if hasattr(model, 'material_balance_constraint'):
        return
    
    # recycle_ratio와 low_carbon_ratio가 모두 있는지 확인
    if not hasattr(model, 'recycle_ratio') or not hasattr(model, 'low_carbon_ratio'):
        print("Type A 자재 제약조건 추가 실패: recycle_ratio 또는 low_carbon_ratio 변수 없음")
        return
    
    # Type A의 경우 비율이 고정값이므로 단순히 체크만 수행
    recycle_ratio_value = model.recycle_ratio
    low_carbon_ratio_value = model.low_carbon_ratio
    
    # 단순히 출력만 수행
    print(f"Type A 자재 제약조건 체크: recycle_ratio={recycle_ratio_value}, low_carbon_ratio={low_carbon_ratio_value}")
    
    # 합이 1을 넘지 않는지 체크 (이미 변수 선언 시 고정값으로 설정되었으므로 여기선 출력만 함)
    total_ratio = recycle_ratio_value + low_carbon_ratio_value
    if total_ratio > 1.0:
        print(f"경고: Type A 비율 합({total_ratio})이 1.0을 초과합니다!")
    else:
        print(f"Type A 비율 합({total_ratio}) 확인 완료")


def is_type_a_cathode(model: ConcreteModel) -> bool:
    """
    모델이 Type A 양극재 구성인지 확인
    
    Args:
        model: Pyomo 모델
        
    Returns:
        bool: Type A 양극재 구성인 경우 True
    """
    # recycle_ratio와 low_carbon_ratio가 모델에 속성으로 존재하고 Var 타입이 아닌 경우
    if hasattr(model, 'recycle_ratio') and hasattr(model, 'low_carbon_ratio'):
        from pyomo.environ import Var, Expression
        # 둘 다 변수(Var)가 아니면 Type A로 간주
        if not isinstance(model.recycle_ratio, Var) and not isinstance(model.low_carbon_ratio, Var):
            # Expression도 아닌지 체크
            if not isinstance(model.recycle_ratio, Expression) and not isinstance(model.low_carbon_ratio, Expression):
                return True
    
    return False