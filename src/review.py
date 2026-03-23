#!/usr/bin/env python3
"""AI Code Review Gate — evaluates PR code quality via LLM."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import requests


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass
class Config:
    api_key: str
    threshold: int
    model: str
    custom_rules: Optional[str]
    api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    max_diff_length: int = 12_000
    timeout: int = 30
    rules_timeout: int = 10

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            sys.exit("❌ GROQ_API_KEY is not set.")
        return cls(
            api_key=api_key,
            threshold=int(os.getenv("AI_THRESHOLD", "70")),
            model=os.getenv("AI_MODEL", "llama-3.3-70b-versatile"),
            custom_rules=os.getenv("AI_CUSTOM_RULES") or None,
        )


# ==============================================================================
# Rules & Prompts
# ==============================================================================

DEFAULT_RULES = """\
1. Security (30 pts) — injections, exposed secrets, vulnerabilities.
2. Code quality (25 pts) — style, readability, complexity.
3. Logic (25 pts) — architecture, error handling, SOLID principles.
4. Tests (20 pts) — coverage of new functionality.

Requirements:
- Return strict JSON with no markdown fences.
- Be strict: assign < 50 for critical errors.
- If there are no changes — assign 100.
"""

ANGULAR_RULES = """\
ANGULAR CODING STANDARDS (FULL COMPACT)
=======================================

[ANGULAR CORE]
1. Selector prefix: service-based (cp-, cc-, cb-)
2. changeDetection: OnPush (all components)
3. standalone: true (default)
4. ❌ template/styles in @Component decorator
5. Pipes: pure: true (default)
6. ❌ any type | ❌ eslint-ignore
7. Components = independent logical units
8. Prefer latest Angular API (current version)
9. ❌ public/protected injects (use private)

[NAMING]
1. Full/clear names or obvious abbreviations only
    ❌ val/updateVal(a) → ✅ name/updateName(newName)
2. Methods start with verb
    ❌ nameDenis()/nameUpdatedWithNewName() → ✅ initName()/updateName()
3. Booleans: is/are prefix
    ❌ ready/participantsExists → ✅ isReady/areParticipantsExists
4. Default values priority for props/vars
    ❌ entity!:Entity; exists!:boolean → ✅ entity:Entity|null=null; exists=false

[GENERAL LOGIC]
1. ❌ Logic inside if-blocks → extract methods (SRP)
    ❌ if(a){if(b){...}} → ✅ if(!a)return; ... doX(); if(!b)return; doY();
2. Declare vars at usage point only
    ❌ const a=fn1(),b=fn2(),e=createEntity(a,b);return e;
    ✅ return createEntity(fn1(),fn2());
3. Strict equality === only (never ==)

[PROPERTIES/METHODS]
1. Always explicit access modifier (public/protected/private)
    ❌ _weight=...; increaseWeight(){} → ✅ private readonly _weight=...; public increaseWeight(){}
2. Prefer readonly for properties
    ❌ private _w=...; public weight=this._w.asObservable(); public name=this._name;
    ✅ private readonly _w=...; public readonly weight=this._w.asObservable(); public get name(){return this._name;}
3. Explicit types where TS can't infer
    ❌ weapon=input.required(); name=input<string>('Name');
    ✅ weapon=input.required<Weapon>(); name=input('Name'); // inferred
4. Methods: explicit return type + param types
    ❌ increaseWeight(w){...} calculateWeight(w){...}
    ✅ increaseWeight(w:number):void{...} calculateWeight(w:number):number{...}
5. Lifecycle-internal methods: private + lifecycle prefix
    ❌ public setDefaultWeight(){} called from ngOnInit
    ✅ private initWeight(){} called from ngOnInit

[TEMPLATE METHODS]
1. Event handlers: on* prefix
    ❌ (click)="increaseWeight()" → ✅ (click)="onIncreaseWeight()"
2. ❌ Methods in template (except signals/@cuiPure)
    ❌ {{getWeight()}} → ✅ {{weight()}} via computed(() => this._weight()*Math.random())

[PROPERTY ORDER - CLASSES]
Access hierarchy: private[readonly] → protected[readonly] → public[readonly]
Within each access level:
    1. inject  2. CONSTANTS  3. default vars  4. signals
    5. viewChild/contentChild  6. input/output  7. getters/setters

[METHOD ORDER - CLASSES]
Access: public → protected → private
Within each: 1.Lifecycle(ngOnInit) → 2.on* handlers → 3.lifecycle-prefixed(init*) → 4.others

[STYLES STANDARD]
1. ❌ !important
2. Colors ONLY from @cuby-ui variables
    ❌ color:#888781; → ✅ color:$cui-base-500;
3. Padding/shadow/font/flex ONLY via @cuby-ui mixins
    ❌ display:flex;gap:8px; → ✅ @include cui-flex(8px);

[SELECTORS - STYLES]
1. Selectors: brief, no abbreviations
2. Prefer class selectors
    ❌ header{} #button{} → ✅ .header{} .button{}
3. Modifiers via _ (underscore): &_active
4. Nested elements via __ (double underscore): &__title
5. Nesting strictly via & syntax

[TEMPLATE STANDARD]
1. Avoid wrapper hell — nesting only where needed
2. Boolean attributes without value: ✅ required checked
3. Binding over interpolation: ✅ [title]="'Text'"
4. Transloco: single directive per template, not pipes

[ATTRIBUTE SEQUENCE - TEMPLATE]
1. HTML attrs (type/autofocus)
2. Directives (cuiButton)
3. Inputs ([disabled])
4. Classes/Styles (class/style)
5. Outputs ((click))
"""

SYSTEM_PROMPT = (
    "You are a senior technical code reviewer. "
    "Evaluate the provided diff and return ONLY valid JSON: "
    '{"score": <0-100>, "comment": "<review text>"}.'
)

REVIEW_PROMPT_TEMPLATE = """\
Use the following rules for evaluation:

{rules}

GIT DIFF:
{diff}
"""


# ==============================================================================
# Rule Engine
# ==============================================================================


class RuleEngine:
    """Manages evaluation rules — default, inline text, or fetched from a URL."""

    def __init__(self, custom_rules: Optional[str] = None, timeout: int = 10) -> None:
        self._custom_rules = custom_rules
        self._timeout = timeout

    @staticmethod
    def _is_url(value: str) -> bool:
        return value.startswith(("http://", "https://"))

    def _fetch_url(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=self._timeout)
            response.raise_for_status()
            return response.text.strip()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to load rules from URL: {exc}") from exc

    def get_rules(self) -> str:
        if not self._custom_rules:
            return DEFAULT_RULES
        if self._is_url(self._custom_rules):
            return self._fetch_url(self._custom_rules)
        if os.path.isfile(self._custom_rules):
            with open(self._custom_rules) as fh:
                return fh.read().strip()
        return self._custom_rules

    def build_prompt(self, diff: str) -> str:
        rules = f"{self.get_rules()}\n{ANGULAR_RULES}"
        return REVIEW_PROMPT_TEMPLATE.format(rules=rules, diff=diff)

    @property
    def source_label(self) -> str:
        if not self._custom_rules:
            return "Default"
        if self._is_url(self._custom_rules):
            return f"URL: {self._custom_rules}"
        if os.path.isfile(self._custom_rules):
            return f"File: {self._custom_rules}"
        return "Custom (inline)"


# ==============================================================================
# Code Reviewer
# ==============================================================================


class CodeReviewer:
    def __init__(self, config: Config, rule_engine: RuleEngine) -> None:
        self._config = config
        self._rule_engine = rule_engine
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Git diff
    # ------------------------------------------------------------------

    def get_diff(self) -> str:
        """Returns the full PR diff (all commits, not just the last one)."""
        if not os.path.exists(".git"):
            print(
                "⚠️  .git directory not found. "
                "Run actions/checkout before this step.",
                file=sys.stderr,
            )
            return ""

        # Strategy 1 — GitHub event payload: most reliable source for PR base SHA.
        # Always available in GitHub Actions; unaffected by remote tracking ref state.
        base_sha = self._base_sha_from_event()
        if base_sha:
            diff = self._run_diff(["git", "diff", base_sha, "HEAD"])
            if diff is not None:
                print(f"📊 Diff: {len(diff):,} chars from base commit {base_sha[:7]}")
                return diff

        # Strategy 2 — remote tracking branch (origin/<base_ref>).
        base_ref = os.getenv("GITHUB_BASE_REF", "main")
        remote_ref = f"origin/{base_ref}"

        if not self._ref_exists(remote_ref):
            print(f"⚠️  {remote_ref} not found locally, fetching...", file=sys.stderr)
            self._fetch_branch(base_ref)

        if self._ref_exists(remote_ref):
            diff = self._run_diff(["git", "diff", f"{remote_ref}...HEAD"])
            if diff is not None:
                print(f"📊 Diff: {len(diff):,} chars against {remote_ref}")
                return diff

        # Strategy 3 — last commit only (last resort, covers single-commit PRs).
        if self._ref_exists("HEAD~1"):
            print("⚠️  Falling back to HEAD~1..HEAD (last commit only).", file=sys.stderr)
            return self._run_diff(["git", "diff", "HEAD~1", "HEAD"]) or ""

        print("⚠️  Cannot determine diff range.", file=sys.stderr)
        return ""

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    def evaluate(self, diff: str) -> Tuple[int, str]:
        """Sends the diff to the LLM and returns (score, comment)."""
        if not diff:
            return 100, "No changes detected."

        prompt = self._rule_engine.build_prompt(diff[: self._config.max_diff_length])

        try:
            response = self._session.post(
                self._config.api_url,
                json={
                    "model": self._config.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
                timeout=self._config.timeout,
            )
            if not response.ok:
                try:
                    api_msg = response.json().get("error", {}).get("message", response.text)
                except Exception:
                    api_msg = response.text
                return 0, f"API error {response.status_code}: {api_msg}"

            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(
                content.replace("```json", "").replace("```", "").strip()
            )
            score = max(0, min(100, int(result.get("score", 0))))
            comment = result.get("comment", "No comment returned.")
            return score, comment

        except Exception as exc:  # noqa: BLE001
            return 0, f"Analysis error: {exc}"

    # ------------------------------------------------------------------
    # GitHub Actions output
    # ------------------------------------------------------------------

    def save_output(self, score: int, comment: str) -> None:
        """Writes step outputs to $GITHUB_OUTPUT."""
        output_file = os.getenv("GITHUB_OUTPUT")
        if not output_file:
            return
        with open(output_file, "a") as fh:
            fh.write(f"ai_score={score}\n")
            fh.write(f"ai_comment={comment.replace(chr(10), '%0A')}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _base_sha_from_event() -> Optional[str]:
        """Reads the PR base commit SHA from the GitHub Actions event payload."""
        event_path = os.getenv("GITHUB_EVENT_PATH", "")
        if not event_path or not os.path.exists(event_path):
            return None
        try:
            with open(event_path) as fh:
                event = json.load(fh)
            sha = event.get("pull_request", {}).get("base", {}).get("sha")
            return sha or None
        except Exception:
            return None

    @staticmethod
    def _ref_exists(ref: str) -> bool:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    @staticmethod
    def _fetch_branch(branch: str) -> None:
        result = subprocess.run(
            ["git", "fetch", "origin", branch],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            snippet = result.stderr[:300].strip()
            print(f"⚠️  git fetch failed: {snippet}", file=sys.stderr)

    @staticmethod
    def _run_diff(cmd: list) -> str:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Truncate to avoid dumping git's full help text into the log.
            snippet = result.stderr[:300].strip()
            print(f"⚠️  git diff failed: {snippet}", file=sys.stderr)
            return ""
        return result.stdout


# ==============================================================================
# Entry point
# ==============================================================================


def main() -> None:
    config = Config.from_env()
    rule_engine = RuleEngine(config.custom_rules, config.rules_timeout)
    reviewer = CodeReviewer(config, rule_engine)

    print(f"🚀 AI Review | Model: {config.model} | Threshold: {config.threshold}")
    print(f"📜 Rules: {rule_engine.source_label}")
    print("-" * 50)

    diff = reviewer.get_diff()
    if not diff:
        print(
            "⚠️  Empty git diff. "
            "Make sure actions/checkout runs with fetch-depth: 0.",
            file=sys.stderr,
        )

    score, comment = reviewer.evaluate(diff)

    print(f"🎯 Score:   {score}/100")
    print(f"💬 Comment: {comment}")
    print("-" * 50)

    reviewer.save_output(score, comment)

    if score < config.threshold:
        print(f"❌ FAILED ({score} < {config.threshold})")
        sys.exit(1)

    print("✅ PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
