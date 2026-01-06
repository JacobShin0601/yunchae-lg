import os
import json
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import threading

@dataclass
class TokenUsage:
    """개별 호출의 토큰 사용량"""
    timestamp: str
    agent_name: str
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_used: bool = False
    tool_calls: int = 0
    reasoning_tokens: int = 0

@dataclass
class AgentStats:
    """에이전트별 통계"""
    agent_name: str
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost: float
    models_used: List[str]
    cache_hits: int
    tool_calls: int

class TokenTracker:
    """실시간 토큰 사용량 추적 관리자"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.usage_history: List[TokenUsage] = []
        self.session_start = datetime.now()
        self.current_session_id = f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}"
        
        # Bedrock 요금표 (USD per 1K tokens)
        self.pricing = {
            "anthropic.claude-3-7-sonnet-20250219-v1:0": {
                "input": 0.003,
                "output": 0.015,
                "cache_write": 0.00375,
                "cache_read": 0.0003
            },
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {
                "input": 0.003,
                "output": 0.015,
                "cache_write": 0.00375,
                "cache_read": 0.0003
            },
            "anthropic.claude-3-5-sonnet-20240620-v1:0": {
                "input": 0.003,
                "output": 0.015,
                "input_cache": 0.0015,
                "output_cache": 0.0075,
                "cache_write": 0.00375,
                "cache_read": 0.0003
            },
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
                "input": 0.003,
                "output": 0.015,
                "input_cache": 0.0015,
                "output_cache": 0.0075,
                "cache_write": 0.00375,
                "cache_read": 0.0003
            },
            "anthropic.claude-3-5-haiku-20241022-v1:0": {
                "input": 0.0008,
                "output": 0.004,
                "input_cache": 0.0005,
                "output_cache": 0.0025,
                "cache_write": 0.001,
                "cache_read": 0.00008
            },
            "us.amazon.nova-pro-v1:0": {
                "input": 0.0008,
                "output": 0.0032,
                "cache_write": 0.0010,
                "cache_read": 0.00008
            }
        }
        
        # 에이전트별 모델 매핑 (agents.py에서 가져온 정보)
        self.agent_model_map = {
            "coordinator": "claude_3_5_sonnet",
            "planner": "claude_3_7_sonnet", 
            "supervisor": "claude_3_7_sonnet",
            "researcher": "claude_3_7_sonnet",
            "coder": "claude_3_7_sonnet", 
            "reporter": "claude_3_7_sonnet"
        }
        
        # 캐시 사용 여부 (config/agents.py 기반)
        self.agent_cache_map = {
            "coordinator": False,
            "planner": True,
            "supervisor": True, 
            "researcher": False,
            "coder": False,
            "reporter": True
        }
    
    def track_usage(self, agent_name: str, model_id: str, token_usage: Dict, 
                   cache_used: bool = False, tool_calls: int = 0, reasoning_tokens: int = 0):
        """토큰 사용량 추적"""
        usage = TokenUsage(
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            model_id=model_id,
            input_tokens=token_usage.get('inputTokens', 0),
            output_tokens=token_usage.get('outputTokens', 0),
            total_tokens=token_usage.get('totalTokens', 0),
            cache_used=cache_used,
            tool_calls=tool_calls,
            reasoning_tokens=reasoning_tokens
        )
        
        self.usage_history.append(usage)
        
        # 실시간 로그
        print(f"📊 Token Usage - {agent_name}: {usage.input_tokens}→{usage.output_tokens} tokens (${self._calculate_cost(usage):.6f})")
    
    def _calculate_cost(self, usage: TokenUsage) -> float:
        """개별 호출의 비용 계산"""
        if usage.model_id not in self.pricing:
            return 0.0
        
        pricing = self.pricing[usage.model_id]
        
        if usage.cache_used:
            # 캐시 사용 시 50% 히트율 가정
            cache_hit_ratio = 0.5
            input_cost = (usage.input_tokens * cache_hit_ratio * pricing.get("cache_read", 0) / 1000) + \
                        (usage.input_tokens * (1 - cache_hit_ratio) * pricing["input"] / 1000)
            output_cost = usage.output_tokens * pricing.get("output_cache", pricing["output"]) / 1000
            cache_write_cost = usage.input_tokens * 0.1 * pricing.get("cache_write", 0) / 1000
            return input_cost + output_cost + cache_write_cost
        else:
            input_cost = usage.input_tokens * pricing["input"] / 1000
            output_cost = usage.output_tokens * pricing["output"] / 1000
            return input_cost + output_cost
    
    def get_agent_stats(self) -> Dict[str, AgentStats]:
        """에이전트별 통계 계산"""
        stats = {}
        
        for agent_name in set(usage.agent_name for usage in self.usage_history):
            agent_usages = [u for u in self.usage_history if u.agent_name == agent_name]
            
            total_calls = len(agent_usages)
            total_input = sum(u.input_tokens for u in agent_usages)
            total_output = sum(u.output_tokens for u in agent_usages)
            total_tokens = sum(u.total_tokens for u in agent_usages)
            total_cost = sum(self._calculate_cost(u) for u in agent_usages)
            models_used = list(set(u.model_id for u in agent_usages))
            cache_hits = sum(1 for u in agent_usages if u.cache_used)
            tool_calls = sum(u.tool_calls for u in agent_usages)
            
            stats[agent_name] = AgentStats(
                agent_name=agent_name,
                total_calls=total_calls,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_tokens=total_tokens,
                total_cost=total_cost,
                models_used=models_used,
                cache_hits=cache_hits,
                tool_calls=tool_calls
            )
        
        return stats
    
    def get_total_cost(self) -> float:
        """총 비용 계산"""
        return sum(self._calculate_cost(usage) for usage in self.usage_history)
    
    def get_session_summary(self) -> Dict:
        """세션 요약 정보"""
        if not self.usage_history:
            return {"error": "No usage data recorded"}
        
        agent_stats = self.get_agent_stats()
        total_cost = self.get_total_cost()
        session_duration = datetime.now() - self.session_start
        
        return {
            "session_id": self.current_session_id,
            "session_start": self.session_start.isoformat(),
            "session_duration_minutes": session_duration.total_seconds() / 60,
            "total_calls": len(self.usage_history),
            "total_cost": total_cost,
            "agent_stats": {name: asdict(stats) for name, stats in agent_stats.items()},
            "cost_breakdown": self._get_cost_breakdown()
        }
    
    def _get_cost_breakdown(self) -> Dict:
        """비용 세부 분석"""
        agent_stats = self.get_agent_stats()
        sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1].total_cost, reverse=True)
        
        breakdown = {
            "by_agent": [],
            "by_model": {},
            "by_operation": {
                "input_tokens": sum(u.input_tokens for u in self.usage_history),
                "output_tokens": sum(u.output_tokens for u in self.usage_history),
                "cache_operations": sum(1 for u in self.usage_history if u.cache_used),
                "tool_calls": sum(u.tool_calls for u in self.usage_history)
            }
        }
        
        # 에이전트별 비용 순위
        for agent_name, stats in sorted_agents:
            breakdown["by_agent"].append({
                "agent": agent_name,
                "cost": stats.total_cost,
                "percentage": (stats.total_cost / self.get_total_cost()) * 100 if self.get_total_cost() > 0 else 0
            })
        
        # 모델별 사용량
        for usage in self.usage_history:
            model = usage.model_id
            if model not in breakdown["by_model"]:
                breakdown["by_model"][model] = {"calls": 0, "cost": 0}
            breakdown["by_model"][model]["calls"] += 1
            breakdown["by_model"][model]["cost"] += self._calculate_cost(usage)
        
        return breakdown
    
    def save_billing_report(self, filepath: str = "./artifacts/bedrock_billing.txt"):
        """실제 사용량 기반 billing 보고서 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        summary = self.get_session_summary()
        if "error" in summary:
            return f"Error: {summary['error']}"
        
        # 보고서 생성
        current_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")
        
        report_content = f"""# Bedrock 실제 사용량 기반 비용 보고서

생성일시: {current_time}
세션 ID: {summary['session_id']}
분석 대상: Agentic Program (bedrock_manus)
세션 지속시간: {summary['session_duration_minutes']:.1f}분

## 📊 실제 사용량 요약

**총 LLM 호출 횟수**: {summary['total_calls']:,}회
**총 비용**: ${summary['total_cost']:.6f}
**평균 호출당 비용**: ${summary['total_cost']/summary['total_calls']:.6f} (호출 {summary['total_calls']}회 기준)

## 💰 에이전트별 실제 사용량 분석

"""
        
        # 에이전트별 상세 분석
        for agent_data in summary['cost_breakdown']['by_agent']:
            agent_name = agent_data['agent']
            agent_stats = summary['agent_stats'][agent_name]
            
            report_content += f"""### {agent_name}
- **총 호출 횟수**: {agent_stats['total_calls']:,}회
- **입력 토큰**: {agent_stats['total_input_tokens']:,}개
- **출력 토큰**: {agent_stats['total_output_tokens']:,}개
- **총 토큰**: {agent_stats['total_tokens']:,}개
- **총 비용**: ${agent_stats['total_cost']:.6f} ({agent_data['percentage']:.1f}%)
- **캐시 히트**: {agent_stats['cache_hits']}회
- **도구 호출**: {agent_stats['tool_calls']}회
- **사용 모델**: {', '.join(agent_stats['models_used'])}

"""
        
        # 모델별 사용량
        report_content += "## 🛠 모델별 사용량\n\n"
        for model, data in summary['cost_breakdown']['by_model'].items():
            model_name = model.split('.')[-1] if '.' in model else model
            report_content += f"- **{model_name}**: {data['calls']}회 호출, ${data['cost']:.6f}\n"
        
        # 운영 통계
        ops = summary['cost_breakdown']['by_operation']
        report_content += f"""
## 📈 운영 통계

- **총 입력 토큰**: {ops['input_tokens']:,}개
- **총 출력 토큰**: {ops['output_tokens']:,}개
- **캐시 사용 횟수**: {ops['cache_operations']}회
- **도구 호출 횟수**: {ops['tool_calls']}회

## 💡 비용 최적화 제안

"""
        
        # 최적화 제안
        if summary['total_cost'] > 0:
            top_agent = summary['cost_breakdown']['by_agent'][0]
            if top_agent['percentage'] > 50:
                report_content += f"- **{top_agent['agent']} 최적화**: 전체 비용의 {top_agent['percentage']:.1f}%를 차지하므로 우선 최적화 대상\n"
            
            if ops['cache_operations'] < summary['total_calls'] * 0.3:
                report_content += "- **캐시 활용 증대**: 캐시 사용률이 낮습니다. 프롬프트 캐싱 활용도를 높이세요\n"
            
            avg_output_ratio = ops['output_tokens'] / ops['input_tokens'] if ops['input_tokens'] > 0 else 0
            if avg_output_ratio > 0.8:
                report_content += "- **출력 길이 최적화**: 출력 토큰 비율이 높습니다. 응답 길이 최적화를 고려하세요\n"
        
        report_content += f"""
## 📋 세부 호출 내역

"""
        
        # 최근 10개 호출 내역
        recent_calls = self.usage_history[-10:] if len(self.usage_history) > 10 else self.usage_history
        for i, usage in enumerate(recent_calls, 1):
            cost = self._calculate_cost(usage)
            timestamp = datetime.fromisoformat(usage.timestamp).strftime("%H:%M:%S")
            report_content += f"{i:2d}. [{timestamp}] {usage.agent_name}: {usage.input_tokens}→{usage.output_tokens} tokens (${cost:.6f})\n"
        
        if len(self.usage_history) > 10:
            report_content += f"\n... 총 {len(self.usage_history)}개 호출 중 최근 10개만 표시\n"
        
        report_content += f"""

---

*이 보고서는 bedrock_manus 프로젝트의 실제 토큰 사용량을 기반으로 생성되었습니다.*
*세션 ID: {summary['session_id']}*
"""
        
        # 파일 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return f"실제 사용량 기반 billing 보고서가 {filepath}에 저장되었습니다."
    
    def reset_session(self):
        """새 세션 시작"""
        self.usage_history.clear()
        self.session_start = datetime.now()
        self.current_session_id = f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}"

# 전역 인스턴스
token_tracker = TokenTracker() 