#!/usr/bin/env python3
"""
AI Code Review Gate
Оценивает качество кода в PR через LLM с поддержкой кастомных правил и URL.
"""

import os
import sys
import json
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

import requests


# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================

@dataclass
class Config:
    api_key: str
    threshold: int
    custom_rules: Optional[str]
    api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    max_diff_length: int = 12000
    timeout: int = 30
    rules_timeout: int = 10

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            api_key=sys.argv[1] if len(sys.argv) > 1 else os.getenv("GROQ_API_KEY", ""),
            threshold=int(sys.argv[2]) if len(sys.argv) > 2 else 70,
            custom_rules=sys.argv[3] if len(sys.argv) > 3 else None
        )


# ==============================================================================
# ПРАВИЛА И ПРОМПТЫ
# ==============================================================================

DEFAULT_RULES = """
1. Безопасность (30 баллов) — инъекции, секреты, уязвимости.
2. Качество кода (25 баллов) — стиль, читаемость, сложность.
3. Логика (25 баллов) — архитектура, обработка ошибок, SOLID.
4. Тесты (20 баллов) — покрытие новой функциональности.

Требования:
- Верни строго JSON без markdown-разметки.
- Будь строгим: за критичные ошибки ставь < 50 баллов.
- Если изменений нет — ставь 100.
"""

SYSTEM_PROMPT = (
    "Ты — старший технический ревьюер. "
    "Твоя задача — оценить код и вернуть ТОЛЬКО валидный JSON: "
    '{"score": 0-100, "comment": "текст на русском"}.'
)

REVIEW_PROMPT_TEMPLATE = """
Используй следующие правила для оценки:

{rules}

GIT DIFF:
{diff}
"""


# ==============================================================================
# ЛОГИКА
# ==============================================================================

class RuleEngine:
    """Управляет правилами оценки (дефолтные, текст или URL)."""
    
    def __init__(self, custom_rules: Optional[str] = None, timeout: int = 10):
        self.custom_rules = custom_rules
        self.timeout = timeout
    
    def _is_url(self, value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")
    
    def _fetch_rules_from_url(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text.strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Не удалось загрузить правила из URL: {str(e)}")
    
    def get_rules(self) -> str:
        if not self.custom_rules:
            return DEFAULT_RULES
        
        if self._is_url(self.custom_rules):
            return self._fetch_rules_from_url(self.custom_rules)
        
        return self.custom_rules
    
    def build_prompt(self, diff: str) -> str:
        rules = self.get_rules()
        return REVIEW_PROMPT_TEMPLATE.format(rules=rules, diff=diff)


class CodeReviewer:
    def __init__(self, config: Config, rule_engine: RuleEngine):
        self.config = config
        self.rule_engine = rule_engine
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        })

    def debug_git():
        cmds = [
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            ["git", "branch", "-a"],
            ["git", "log", "--oneline", "-3"],
            ["git", "remote", "-v"],
        ]
        print("\n🔍 GIT DEBUG INFO:", file=sys.stderr)
        for cmd in cmds:
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"{' '.join(cmd)}:\n{res.stdout}", file=sys.stderr)
            except Exception as e:
                print(f"❌ {' '.join(cmd)}: {e}", file=sys.stderr)
        print("-" * 50, file=sys.stderr)

    def get_diff(self) -> str:
        self.debug_git()

        """Получает git diff текущего PR."""
        base_ref = os.getenv("GITHUB_BASE_REF", "main")
        try:
            result = subprocess.run(
                ["git", "diff", f"origin/{base_ref}...HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    def evaluate(self, diff: str) -> Tuple[int, str]:
        """Отправляет код на оценку и возвращает (балл, комментарий)."""
        if not diff:
            return 100, "Изменения отсутствуют"

        prompt = self.rule_engine.build_prompt(diff[:self.config.max_diff_length])

        try:
            response = self.session.post(
                self.config.api_url,
                json={
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            result = json.loads(
                content.replace("```json", "").replace("```", "").strip()
            )
            score = max(0, min(100, int(result.get("score", 0))))
            comment = result.get("comment", "Нет комментария")
            return score, comment

        except Exception as e:
            return 0, f"Ошибка анализа: {str(e)}"

    def save_output(self, score: int, comment: str) -> None:
        """Сохраняет результаты для GitHub Actions."""
        output_file = os.getenv("GITHUB_OUTPUT")
        if not output_file:
            return

        with open(output_file, "a") as f:
            f.write(f"ai_score={score}\n")
            f.write(f"ai_comment={comment}\n")


# ==============================================================================
# ТОЧКА ВХОДА
# ==============================================================================

def main() -> None:
    config = Config.from_env()
    rule_engine = RuleEngine(config.custom_rules, config.rules_timeout)
    reviewer = CodeReviewer(config, rule_engine)

    print(f"🚀 AI Review | Порог: {config.threshold}")
    
    # Отображение источника правил
    if not config.custom_rules:
        rules_source = "По умолчанию"
    elif rule_engine._is_url(config.custom_rules):
        rules_source = f"URL: {config.custom_rules}"
    else:
        rules_source = "Кастомные (inline)"
    
    print(f"📜 Правила: {rules_source}")
    print("-" * 50)

    diff = reviewer.get_diff()
    score, comment = reviewer.evaluate(diff)

    print(f"🎯 Балл: {score}/100")
    print(f"💬 Комментарий: {comment}")
    print("-" * 50)

    reviewer.save_output(score, comment)

    if score < config.threshold:
        print(f"❌ FAILED ({score} < {config.threshold})")
        sys.exit(1)
    
    print("✅ PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
