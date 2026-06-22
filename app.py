import os
import sys
import json
import random
import urllib.request

# Ensure the src directory is on the Python path so that the ichingshifa
# package can be imported when the app is launched from the project root
# (e.g. ``streamlit run app.py``).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import streamlit.components.v1 as components
import pendulum as pdlm
from contextlib import contextmanager, redirect_stdout
from io import StringIO

from ichingshifa import ichingshifa
from ichingshifa.cerebras_client import (
    CerebrasClient,
    OpenAICompatibleClient,
    DEFAULT_MODEL,
    CEREBRAS_MODEL_OPTIONS,
    CEREBRAS_MODEL_DESCRIPTIONS,
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPTS_FILE = os.path.join(BASE_DIR, "system_prompts.json")
DEFAULT_MAX_TOKENS = 200000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CUSTOM_SERVER = "https://api.openai.com/v1"
DEFAULT_CUSTOM_MODEL = "gpt-4o-mini"

YAO_LABELS = {"6": "老陰", "7": "少陽", "8": "少陰", "9": "老陽"}
YAO_SYMBOLS = {"6": "▅▅ ▅▅ X", "7": "▅▅▅▅▅  ", "8": "▅▅ ▅▅  ", "9": "▅▅▅▅▅ O"}
YAO_POSITIONS = ["初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]

XI_CI_QUOTE = (
    "「大衍之數五十，其用四十有九。分而為二以象兩，掛一以象三，"
    "揲之以四以象四，歸奇於扐以象閏。五歲再閏，故再扐而後掛。」"
    "——《周易·繫辭傳》"
)


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


def read_local_file(path):
    """Read a text file relative to the project root."""
    full = os.path.join(BASE_DIR, path)
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


def get_remote_file(url):
    """Fetch a remote text file."""
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def get_ai_provider_settings():
    provider = st.session_state.get("ai_provider_selector", "Cerebras")
    if provider == "Cerebras":
        api_key = (
            st.session_state.get("cerebras_api_key_input", "").strip()
            or st.secrets.get("CEREBRAS_API_KEY", "")
            or os.getenv("CEREBRAS_API_KEY", "")
        )
        model = st.session_state.get("cerebras_model_selector", DEFAULT_MODEL)
        if not api_key:
            raise ValueError("CEREBRAS API Key 未設置，請在側邊欄或 secrets/環境變量中設置。")
        return provider, model, CerebrasClient(api_key=api_key)

    api_key = st.session_state.get("custom_ai_api_key_input", "").strip()
    server = st.session_state.get("custom_ai_server_input", "").strip()
    model = st.session_state.get("custom_ai_model_input", "").strip()
    if not api_key:
        raise ValueError("請輸入自定義 AI 的 API Key。")
    if not server:
        raise ValueError("請輸入自定義 AI 的 Server URL。")
    if not model:
        raise ValueError("請輸入自定義 AI 的模型名稱。")
    return provider, model, OpenAICompatibleClient(api_key=api_key, base_url=server)


def request_ai_response(messages, max_tokens, temperature):
    provider, model, client = get_ai_provider_settings()
    api_params = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if provider == "Cerebras":
        response = client.get_chat_completion(**api_params)
        return response.choices[0].message.content

    response = client.get_chat_completion(**api_params)
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("AI 未返回任何回應。")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise ValueError("AI 回應內容為空。")
    return content


# ---------------------------------------------------------------------------
# System prompt persistence
# ---------------------------------------------------------------------------

def load_system_prompts():
    """Load system prompts from JSON file, creating defaults if needed."""
    default_content = (
        "你是一位精通周易六爻的大師，熟悉《增刪卜易》、《卜筮正宗》、《黃金策》及歷史占例。"
        "請根據提供的六爻排盤數據，進行以下操作：\n\n"
        "1. 解釋卦象的關鍵要素（本卦、之卦、世應、六親、動爻、伏神等）。\n"
        "2. 結合六爻理論，分析卦象的吉凶和潛在影響。\n"
        "3. 根據用神、原神、忌神、仇神的旺衰及動靜，詳細評估事情的發展趨勢。\n"
        "4. 提供實用的建議或應對策略。\n\n"
        "請以清晰的結構（分段、標題）呈現，語言專業且易懂，適當引用經典理論或歷史占例。"
    )
    try:
        with open(SYSTEM_PROMPTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        default_data = {
            "prompts": [{"name": "六爻大師", "content": default_content}],
            "selected": "六爻大師",
        }
        with open(SYSTEM_PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_data, f, indent=2, ensure_ascii=False)
        return default_data


def save_system_prompts(prompts_data):
    """Persist system prompts to JSON file."""
    try:
        with open(SYSTEM_PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(prompts_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"儲存提示時發生錯誤：{e}")
        return False


# ---------------------------------------------------------------------------
# 大衍筮法狀態機
# ---------------------------------------------------------------------------
#
# 狀態機設計：
#   phase: idle → active → completed
#   yao: 1..6（當前爻位，由下而上）
#   change: 1..3（當前爻內之變）
#   step: divide → hang_one → count_four → change_done → (下一變或 yao_done)
#
# 每爻獨立以 49 策起算；三變後餘策 ÷ 4 = 爻值 (6/7/8/9)

def init_dayan_state():
    defaults = {
        "dayan_phase": "idle",
        "dayan_yao": 0,
        "dayan_change": 0,
        "dayan_step": "ready",
        "dayan_stalks": 49,
        "dayan_changes_data": [],
        "dayan_lines": [],
        "dayan_left": 24,
        "dayan_right": 25,
        "dayan_hung": 0,
        "dayan_left_rem": 0,
        "dayan_right_rem": 0,
        "dayan_removed": 0,
        "dayan_pan_text": "",
        "dayan_combine": "",
        "dayan_show_pan": False,
        "dayan_shuffle": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def remainder_four(n):
    r = n % 4
    return 4 if r == 0 else r


def random_split(stalks):
    lo = max(1, stalks // 3)
    hi = max(lo + 1, (stalks * 2) // 3)
    left = random.randint(lo, hi)
    return left, stalks - left


DAYAN_STAGE_HEIGHT = 208

DAYAN_FONT = '"Noto Serif TC", "Songti SC", "STSong", serif'
DAYAN_MONO = '"Noto Serif TC", "Courier New", monospace'

DAYAN_SCENE_CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
.dayan-scene {{
    font-family: {DAYAN_FONT};
    background:
        radial-gradient(ellipse at 50% 0%, rgba(201,168,76,0.08) 0%, transparent 60%),
        linear-gradient(165deg, #1a1612 0%, #221c16 50%, #181410 100%);
    border: 1px solid rgba(201,168,76,0.25);
    border-radius: 14px;
    padding: 20px 22px;
    color: #d8c8a8;
    position: relative;
    overflow: hidden;
}}
.dayan-scene::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='40' height='40' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 20h40M20 0v40' stroke='%23c9a84c' stroke-width='0.3' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
}}
.dayan-scene .phase {{
    text-align: center;
    font-size: 12px;
    color: #a89060;
    margin-bottom: 16px;
    letter-spacing: 3px;
    position: relative;
}}
.dayan-scene .split-wrap {{
    display: flex;
    align-items: stretch;
    gap: 0;
    height: 88px;
    position: relative;
}}
.dayan-scene .pile-side {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    border-radius: 8px;
    transition: flex-grow 0.5s cubic-bezier(.4,0,.2,1);
    min-width: 56px;
    position: relative;
    overflow: hidden;
}}
.dayan-scene .pile-side::after {{
    content: "";
    position: absolute;
    bottom: 8px; left: 12px; right: 12px;
    height: 3px;
    background: repeating-linear-gradient(90deg, #8b7355 0px, #8b7355 2px, transparent 2px, transparent 5px);
    opacity: 0.35;
    border-radius: 2px;
}}
.dayan-scene .pile-side.left {{
    background: linear-gradient(160deg, rgba(90,70,40,0.35) 0%, rgba(40,34,24,0.6) 100%);
    border: 1px solid rgba(201,168,76,0.2);
    margin-right: 3px;
}}
.dayan-scene .pile-side.right {{
    background: linear-gradient(200deg, rgba(90,70,40,0.35) 0%, rgba(40,34,24,0.6) 100%);
    border: 1px solid rgba(201,168,76,0.2);
    margin-left: 3px;
}}
.dayan-scene .pile-num {{
    font-size: 32px;
    font-weight: 200;
    color: #f0e0c0;
    line-height: 1;
    text-shadow: 0 0 20px rgba(201,168,76,0.3);
}}
.dayan-scene .pile-tag {{
    font-size: 10px;
    color: #7a6e50;
    margin-top: 6px;
    letter-spacing: 2px;
}}
.dayan-scene .split-divider {{
    width: 1px;
    background: linear-gradient(180deg, transparent 5%, #c9a84c 50%, transparent 95%);
    align-self: stretch;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(201,168,76,0.4);
    z-index: 1;
}}
.dayan-scene .hung-chip {{
    display: block;
    width: fit-content;
    margin: 14px auto 0;
    padding: 4px 16px;
    background: rgba(201,168,76,0.1);
    border: 1px solid rgba(201,168,76,0.35);
    border-radius: 20px;
    font-size: 11px;
    color: #d4b060;
    letter-spacing: 2px;
}}
.dayan-scene .count-row {{
    display: flex;
    justify-content: center;
    gap: 48px;
}}
.dayan-scene .count-item {{ text-align: center; }}
.dayan-scene .count-item .n {{
    font-size: 28px;
    font-weight: 200;
    color: #f0e0c0;
}}
.dayan-scene .count-item .t {{
    font-size: 10px;
    color: #7a6e50;
    letter-spacing: 2px;
    margin-top: 4px;
}}
.dayan-scene .count-detail {{
    text-align: center;
    font-size: 11px;
    color: #7a9a6a;
    margin-top: 14px;
    letter-spacing: 1px;
    font-family: {DAYAN_MONO};
}}
.dayan-scene .total {{
    text-align: center;
    font-size: 11px;
    color: #6a5e48;
    margin-top: 14px;
    letter-spacing: 2px;
}}
.dayan-scene.shuffle .pile-side {{
    animation: shuffle 0.5s cubic-bezier(.4,0,.2,1);
}}
@keyframes shuffle {{
    0%   {{ opacity: 1; transform: translateY(0); }}
    35%  {{ opacity: 0.3; transform: translateY(-6px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
"""


def yao_line_html(val):
    if not val:
        return '<div class="yl ghost"><span class="bar"></span></div>'
    yin = val in ("6", "8")
    mark = "×" if val == "6" else ("○" if val == "9" else "")
    cls = "yl yin" if yin else "yl yang"
    if yin:
        inner = '<span class="seg"></span><span class="mid"></span><span class="seg"></span>'
    else:
        inner = '<span class="seg full"></span>'
    m = f'<span class="mk">{mark}</span>' if mark else ""
    return f'<div class="{cls}">{inner}{m}</div>'


HEX_PREVIEW_CSS = f"""
.hex-preview {{
    font-family: {DAYAN_FONT};
    background:
        radial-gradient(ellipse at 50% 100%, rgba(201,168,76,0.06) 0%, transparent 70%),
        linear-gradient(165deg, #1a1612, #201a14);
    border: 1px solid rgba(201,168,76,0.22);
    border-radius: 14px;
    padding: 16px 18px;
    color: #d8c8a8;
}}
.hex-preview .hd {{
    text-align: center;
    font-size: 11px;
    color: #8a7850;
    letter-spacing: 4px;
    margin-bottom: 14px;
}}
.hex-preview .rows {{ display: flex; flex-direction: column; gap: 7px; }}
.hex-preview .row {{
    display: grid;
    grid-template-columns: 36px 1fr 40px;
    align-items: center;
    gap: 8px;
}}
.hex-preview .pos {{
    font-size: 10px;
    color: #5a5040;
    text-align: right;
}}
.hex-preview .lbl {{
    font-size: 10px;
    color: #9a8456;
    text-align: left;
}}
.hex-preview .yl {{
    display: flex;
    align-items: center;
    height: 10px;
    position: relative;
}}
.hex-preview .yl .seg {{
    height: 3px;
    background: #c9a84c;
    border-radius: 1px;
    flex: 1;
    box-shadow: 0 0 6px rgba(201,168,76,0.25);
}}
.hex-preview .yl .seg.full {{ flex: 1; }}
.hex-preview .yl .mid {{ width: 14px; flex-shrink: 0; }}
.hex-preview .yl.ghost .bar {{
    width: 100%; height: 1px;
    background: rgba(201,168,76,0.12);
}}
.hex-preview .yl .mk {{
    position: absolute;
    right: -18px;
    font-size: 9px;
    color: #d4a060;
}}
.hex-preview .names {{
    text-align: center;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid rgba(201,168,76,0.12);
    font-size: 13px;
    color: #d4b060;
    letter-spacing: 1px;
}}
.hex-preview .dots {{
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-top: 12px;
}}
.hex-preview .dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: rgba(201,168,76,0.15);
    border: 1px solid rgba(201,168,76,0.2);
}}
.hex-preview .dot.done {{
    background: #c9a84c;
    box-shadow: 0 0 6px rgba(201,168,76,0.5);
}}
.hex-preview .dot.cur {{
    background: transparent;
    border-color: #d4b060;
    box-shadow: 0 0 8px rgba(212,176,96,0.4);
}}
"""


PROGRESS_PANEL_CSS = f"""
.prog-panel {{
    font-family: {DAYAN_FONT};
    background: linear-gradient(165deg, #1a1612, #201a14);
    border: 1px solid rgba(201,168,76,0.22);
    border-radius: 14px;
    padding: 18px 20px;
    color: #d8c8a8;
    height: 100%;
}}
.prog-panel .title {{
    font-size: 15px;
    color: #d4b060;
    letter-spacing: 6px;
    margin-bottom: 16px;
}}
.prog-panel .stats {{
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
}}
.prog-panel .stat {{
    flex: 1;
    text-align: center;
    padding: 10px 6px;
    background: rgba(201,168,76,0.06);
    border: 1px solid rgba(201,168,76,0.15);
    border-radius: 8px;
}}
.prog-panel .stat .v {{
    font-size: 20px;
    font-weight: 300;
    color: #f0e0c0;
}}
.prog-panel .stat .l {{
    font-size: 10px;
    color: #7a6e50;
    margin-top: 4px;
    letter-spacing: 1px;
}}
.prog-panel .pipeline {{
    display: flex;
    align-items: center;
    gap: 0;
    margin-top: 4px;
}}
.prog-panel .pipe-step {{
    flex: 1;
    text-align: center;
    font-size: 10px;
    color: #5a5040;
    letter-spacing: 1px;
    padding: 6px 2px;
    position: relative;
}}
.prog-panel .pipe-step.active {{
    color: #d4b060;
}}
.prog-panel .pipe-step.done {{
    color: #8a9a6a;
}}
.prog-panel .pipe-step::after {{
    content: "";
    position: absolute;
    bottom: 0; left: 20%; right: 20%;
    height: 2px;
    background: rgba(201,168,76,0.1);
    border-radius: 1px;
}}
.prog-panel .pipe-step.active::after {{
    background: linear-gradient(90deg, transparent, #c9a84c, transparent);
    box-shadow: 0 0 6px rgba(201,168,76,0.4);
}}
.prog-panel .pipe-step.done::after {{
    background: rgba(201,168,76,0.3);
}}
.prog-panel .quote {{
    margin-top: 16px;
    font-size: 11px;
    color: #5a5040;
    line-height: 1.7;
    border-left: 2px solid rgba(201,168,76,0.2);
    padding-left: 10px;
}}
.prog-panel .result {{
    font-size: 22px;
    font-weight: 300;
    color: #f0e0c0;
    letter-spacing: 3px;
    margin-bottom: 6px;
}}
.prog-panel .result-sub {{
    font-size: 12px;
    color: #9a8456;
}}
"""


def render_progress_panel(phase, yao, change, step, lines, combine=""):
    step_order = ["divide", "hang_one", "count_four", "change_done", "yao_done"]
    step_names = ["分堆", "掛一", "揲四", "歸奇", "成爻"]
    cur_idx = step_order.index(step) if step in step_order else 0

    pipe = ""
    for i, name in enumerate(step_names):
        if phase != "active":
            cls = ""
        elif i < cur_idx:
            cls = "done"
        elif i == cur_idx:
            cls = "active"
        else:
            cls = ""
        pipe += f'<div class="pipe-step {cls}">{name}</div>'

    if phase == "idle":
        body = f"""
        <div class="stats">
            <div class="stat"><div class="v">49</div><div class="l">蓍草</div></div>
            <div class="stat"><div class="v">0</div><div class="l">成爻</div></div>
        </div>
        <div class="quote">大衍之數五十，其用四十有九。</div>
        """
        height = 195
    elif phase == "active":
        body = f"""
        <div class="stats">
            <div class="stat"><div class="v">{yao}</div><div class="l">爻 / 6</div></div>
            <div class="stat"><div class="v">{change}</div><div class="l">變 / 3</div></div>
            <div class="stat"><div class="v">{len(lines)}</div><div class="l">已成</div></div>
        </div>
        <div class="pipeline">{pipe}</div>
        """
        height = 210
    else:
        ben, zhi = "", ""
        if combine:
            d = ichingshifa.Iching().mget_bookgua_details(combine)
            ben, zhi = d[1], d[2]
        body = f"""
        <div class="result">{combine}</div>
        <div class="result-sub">{ben} → {zhi}</div>
        """
        height = 160

    html = f"""
    <style>{PROGRESS_PANEL_CSS}</style>
    <div class="prog-panel">
        <div class="title">大衍筮法</div>
        {body}
    </div>
    """
    components.html(html, height=height, scrolling=False)


def render_result_card(title, subtitle="", variant="default"):
    colors = {
        "default": ("#d4b060", "rgba(201,168,76,0.08)"),
        "success": ("#8aba7a", "rgba(120,180,100,0.08)"),
    }
    color, bg = colors.get(variant, colors["default"])
    html = f"""
    <style>
    .rcard-wrap {{
        font-family: {DAYAN_FONT};
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
            radial-gradient(ellipse at 50% 0%, rgba(201,168,76,0.06) 0%, transparent 60%),
            linear-gradient(165deg, #1a1612, #201a14);
        border: 1px solid rgba(201,168,76,0.22);
        border-radius: 14px;
    }}
    .rcard {{
        text-align: center;
        padding: 20px;
        background: {bg};
        border: 1px solid rgba(201,168,76,0.18);
        border-radius: 10px;
        min-width: 60%;
    }}
    .rcard .t {{ font-size: 15px; color: {color}; letter-spacing: 2px; }}
    .rcard .s {{ font-size: 12px; color: #6a5e48; margin-top: 8px; letter-spacing: 1px; }}
    </style>
    <div class="rcard-wrap">
        <div class="rcard">
            <div class="t">{title}</div>
            {"<div class='s'>" + subtitle + "</div>" if subtitle else ""}
        </div>
    </div>
    """
    components.html(html, height=DAYAN_STAGE_HEIGHT, scrolling=False)


def render_completion_stage(combine):
    ben, zhi = "", ""
    if combine:
        d = ichingshifa.Iching().mget_bookgua_details(combine)
        ben, zhi = d[1], d[2]
    html = f"""
    <style>
    .done-wrap {{
        font-family: {DAYAN_FONT};
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background:
            radial-gradient(ellipse at 50% 30%, rgba(201,168,76,0.1) 0%, transparent 70%),
            linear-gradient(165deg, #1a1612, #201a14);
        border: 1px solid rgba(201,168,76,0.22);
        border-radius: 14px;
        color: #d8c8a8;
    }}
    .done-wrap .code {{
        font-size: 26px;
        font-weight: 200;
        color: #f0e0c0;
        letter-spacing: 6px;
    }}
    .done-wrap .name {{
        font-size: 13px;
        color: #9a8456;
        margin-top: 10px;
        letter-spacing: 2px;
    }}
    </style>
    <div class="done-wrap">
        <div class="code">{combine}</div>
        <div class="name">{ben} → {zhi}</div>
    </div>
    """
    components.html(html, height=DAYAN_STAGE_HEIGHT, scrolling=False)


def render_divide_scene(left, right, total, shuffle=False):
    lp = max(left / total * 100, 8)
    rp = max(right / total * 100, 8)
    shuffle_cls = "shuffle" if shuffle else ""
    html = f"""
    <style>{DAYAN_SCENE_CSS}</style>
    <div class="dayan-scene {shuffle_cls}">
        <div class="phase">分而為二 · 共 {total} 策</div>
        <div class="split-wrap">
            <div class="pile-side left" style="flex-grow:{lp}">
                <div class="pile-num">{left}</div>
                <div class="pile-tag">左堆</div>
            </div>
            <div class="split-divider"></div>
            <div class="pile-side right" style="flex-grow:{rp}">
                <div class="pile-num">{right}</div>
                <div class="pile-tag">右堆</div>
            </div>
        </div>
        <div class="total">分而為二以象兩</div>
    </div>
    """
    components.html(html, height=DAYAN_STAGE_HEIGHT, scrolling=False)


def render_yarrow_scene(left, right, hung=0, phase_label="", mode="default"):
    hung_html = (
        f'<div style="text-align:center"><span class="hung-chip">掛一 · {hung}</span></div>'
        if hung else ""
    )
    if mode == "count":
        lg, lr = left // 4, right // 4
        lrem, rrem = remainder_four(left), remainder_four(right)
        detail = f"左 {left} → {lg}×4+{lrem}　·　右 {right} → {lr}×4+{rrem}"
        body = f"""
        <div class="count-row">
            <div class="count-item"><div class="n">{left}</div><div class="t">左堆</div></div>
            <div class="count-item"><div class="n">{right}</div><div class="t">右堆</div></div>
        </div>
        <div class="count-detail">{detail}</div>
        {hung_html}
        """
    else:
        body = f"""
        <div class="count-row">
            <div class="count-item"><div class="n">{left}</div><div class="t">左堆</div></div>
            <div class="count-item"><div class="n">{right}</div><div class="t">右堆</div></div>
        </div>
        {hung_html}
        <div class="total">合計 {left + right + hung} 策</div>
        """

    html = f"""
    <style>{DAYAN_SCENE_CSS}</style>
    <div class="dayan-scene">
        <div class="phase">{phase_label}</div>
        {body}
    </div>
    """
    components.html(html, height=DAYAN_STAGE_HEIGHT, scrolling=False)


def render_hexagram_preview(lines, title="卦象"):
    """lines: list of '6'/'7'/'8'/'9' from bottom to top."""
    padded = list(lines) + [""] * (6 - len(lines))

    zhi_gua = "".join(
        v.replace("6", "7").replace("9", "8") for v in lines
    ) if lines else ""

    ben_name = ""
    zhi_name = ""
    if len(lines) == 6:
        details = ichingshifa.Iching().mget_bookgua_details("".join(lines))
        ben_name = details[1] if details else ""
        zhi_name = details[2] if details else ""

    rows = ""
    for i in range(5, -1, -1):
        val = padded[i]
        rows += (
            f'<div class="row">'
            f'<span class="pos">{YAO_POSITIONS[i]}</span>'
            f'{yao_line_html(val)}'
            f'<span class="lbl">{YAO_LABELS.get(val, "")}</span>'
            f'</div>'
        )

    dots = ""
    for i in range(6):
        if i < len(lines):
            cls = "dot done"
        elif i == len(lines) and lines:
            cls = "dot cur"
        else:
            cls = "dot"
        dots += f'<span class="{cls}"></span>'

    name_line = ""
    if ben_name:
        name_line = ben_name
        if zhi_name and zhi_gua != "".join(lines):
            name_line += f" → {zhi_name}"

    html = f"""
    <style>{HEX_PREVIEW_CSS}</style>
    <div class="hex-preview">
        <div class="hd">{title}</div>
        <div class="rows">{rows}</div>
        <div class="dots">{dots}</div>
        {"<div class='names'>" + name_line + "</div>" if name_line else ""}
    </div>
    """
    h = 280 if ben_name else 250
    components.html(html, height=h, scrolling=False)


def reset_dayan():
    st.session_state.dayan_phase = "idle"
    st.session_state.dayan_yao = 0
    st.session_state.dayan_change = 0
    st.session_state.dayan_step = "ready"
    st.session_state.dayan_stalks = 49
    st.session_state.dayan_changes_data = []
    st.session_state.dayan_lines = []
    st.session_state.dayan_left = 24
    st.session_state.dayan_right = 25
    st.session_state.dayan_hung = 0
    st.session_state.dayan_pan_text = ""
    st.session_state.dayan_combine = ""
    st.session_state.dayan_show_pan = False
    st.session_state.dayan_shuffle = False


def start_yao(yao_num):
    st.session_state.dayan_phase = "active"
    st.session_state.dayan_yao = yao_num
    st.session_state.dayan_change = 1
    st.session_state.dayan_step = "divide"
    st.session_state.dayan_stalks = 49
    st.session_state.dayan_changes_data = []
    left, right = random_split(49)
    st.session_state.dayan_left = left
    st.session_state.dayan_right = right
    st.session_state.dayan_hung = 0
    st.session_state.dayan_shuffle = True


def apply_divide(left_count):
    total = st.session_state.dayan_stalks
    left_count = max(1, min(total - 1, left_count))
    st.session_state.dayan_left = left_count
    st.session_state.dayan_right = total - left_count
    st.session_state.dayan_step = "hang_one"


def apply_hang_one():
    if st.session_state.dayan_right < 1:
        return
    st.session_state.dayan_hung = 1
    st.session_state.dayan_right -= 1
    st.session_state.dayan_step = "count_four"


def apply_count_four():
    left = st.session_state.dayan_left
    right = st.session_state.dayan_right
    left_rem = remainder_four(left)
    right_rem = remainder_four(right)
    removed = left_rem + right_rem + st.session_state.dayan_hung
    st.session_state.dayan_left_rem = left_rem
    st.session_state.dayan_right_rem = right_rem
    st.session_state.dayan_removed = removed
    st.session_state.dayan_changes_data.append({
        "change": st.session_state.dayan_change,
        "left": left,
        "right": right,
        "hung": st.session_state.dayan_hung,
        "left_rem": left_rem,
        "right_rem": right_rem,
        "removed": removed,
    })
    st.session_state.dayan_step = "change_done"


def advance_after_change():
    if st.session_state.dayan_change < 3:
        removed_so_far = sum(c["removed"] for c in st.session_state.dayan_changes_data)
        st.session_state.dayan_stalks = 49 - removed_so_far
        st.session_state.dayan_change += 1
        st.session_state.dayan_step = "divide"
        left, right = random_split(st.session_state.dayan_stalks)
        st.session_state.dayan_left = left
        st.session_state.dayan_right = right
        st.session_state.dayan_hung = 0
        st.session_state.dayan_shuffle = True
    else:
        total_removed = sum(c["removed"] for c in st.session_state.dayan_changes_data)
        remaining = 49 - total_removed
        yao_val = str(remaining // 4)
        st.session_state.dayan_lines.append(yao_val)
        st.session_state.dayan_step = "yao_done"
        if len(st.session_state.dayan_lines) >= 6:
            st.session_state.dayan_phase = "completed"
            st.session_state.dayan_combine = "".join(st.session_state.dayan_lines)


def _render_dayan_stage(phase, step, yao, change, lines, combine, stalks):
    """固定高度內容區，避免按鈕隨步驟跳動。"""
    if phase == "idle":
        render_divide_scene(24, 25, 49)
    elif phase == "active":
        if step == "divide":
            render_divide_scene(
                st.session_state.dayan_left,
                st.session_state.dayan_right,
                stalks,
                shuffle=st.session_state.dayan_shuffle,
            )
            st.session_state.dayan_shuffle = False
        elif step == "hang_one":
            render_yarrow_scene(
                st.session_state.dayan_left,
                st.session_state.dayan_right,
                0,
                f"第{yao}爻 · 第{change}變 · 掛一以象三",
            )
        elif step == "count_four":
            render_yarrow_scene(
                st.session_state.dayan_left,
                st.session_state.dayan_right,
                st.session_state.dayan_hung,
                f"第{yao}爻 · 第{change}變 · 揲之以四",
                mode="count",
            )
        elif step == "change_done":
            last = st.session_state.dayan_changes_data[-1]
            render_result_card(
                f"第 {change} 變　歸奇 {last['removed']} 策",
                f"{last['left_rem']} + {last['right_rem']} + {last['hung']}",
            )
        elif step == "yao_done":
            yao_val = lines[-1]
            render_result_card(
                f"第 {yao} 爻　{YAO_LABELS[yao_val]}",
                YAO_SYMBOLS[yao_val].strip(),
                variant="success",
            )
    elif phase == "completed":
        render_completion_stage(combine)


def _render_dayan_action(phase, step, yao, change, lines, combine):
    """固定位置操作列。"""
    if phase == "idle":
        if st.button("開始起卦", type="primary", use_container_width=True):
            start_yao(1)
            st.rerun()
    elif phase == "active":
        if step == "divide":
            if st.button("分堆", type="primary", use_container_width=True):
                apply_divide(st.session_state.dayan_left)
                st.rerun()
        elif step == "hang_one":
            if st.button("掛一", type="primary", use_container_width=True):
                apply_hang_one()
                st.rerun()
        elif step == "count_four":
            if st.button("揲四", type="primary", use_container_width=True):
                apply_count_four()
                st.rerun()
        elif step == "change_done":
            label = f"第 {change + 1} 變" if change < 3 else "成爻"
            if st.button(label, type="primary", use_container_width=True):
                advance_after_change()
                st.rerun()
        elif step == "yao_done" and yao < 6:
            if st.button(f"第 {yao + 1} 爻", type="primary", use_container_width=True):
                start_yao(yao + 1)
                st.rerun()
    elif phase == "completed":
        c1, c2 = st.columns(2)
        with c1:
            if st.button("生成排盤", type="primary", use_container_width=True):
                st.session_state.dayan_show_pan = True
                st.session_state.classic_combine = combine
                st.session_state.classic_mode = "dayan_manual"
                st.rerun()
        with c2:
            if st.button("重新起卦", use_container_width=True):
                reset_dayan()
                st.rerun()


def render_dayan_tab():
    init_dayan_state()

    phase = st.session_state.dayan_phase
    yao = st.session_state.dayan_yao
    change = st.session_state.dayan_change
    step = st.session_state.dayan_step
    lines = st.session_state.dayan_lines
    combine = st.session_state.dayan_combine
    stalks = st.session_state.dayan_stalks

    col_preview, col_progress = st.columns([3, 2])
    with col_preview:
        render_hexagram_preview(lines)
    with col_progress:
        render_progress_panel(phase, yao, change, step, lines, combine)

    st.markdown('<div class="dayan-stage-marker"></div>', unsafe_allow_html=True)
    _render_dayan_stage(phase, step, yao, change, lines, combine, stalks)
    st.markdown('<div class="dayan-action-marker"></div>', unsafe_allow_html=True)
    _render_dayan_action(phase, step, yao, change, lines, combine)

    if phase == "completed" and st.session_state.dayan_show_pan:
            st.subheader("完整納甲排盤")
            output_pan = st.empty()
            with st.spinner("生成排盤中…"):
                with st_capture(output_pan.code):
                    pan = ichingshifa.Iching().display_pan_m(
                        y, m, d, h, mi, combine
                    )
                    print(pan)
                st.session_state.dayan_pan_text = pan


# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="堅六爻-周易排盤", page_icon="☯️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@200;400&display=swap');

.stApp {
    background: linear-gradient(180deg, #0e0c0a 0%, #14110e 100%);
}
.stCode code, .stCode pre {
    font-size: 15px !important;
    line-height: 1.6 !important;
}
[data-testid="stTabs"] button {
    font-family: "Noto Serif TC", serif;
    letter-spacing: 1px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #c9a84c !important;
    border-bottom-color: #c9a84c !important;
}
div[data-testid="stVerticalBlock"] > div:has(iframe) {
    margin-bottom: 0;
}
.dayan-stage-marker + div {
    height: 208px !important;
    min-height: 208px !important;
    max-height: 208px !important;
    overflow: hidden !important;
    margin-bottom: 0 !important;
}
.dayan-stage-marker + div iframe {
    height: 208px !important;
    min-height: 208px !important;
}
.dayan-action-marker + div {
    min-height: 52px !important;
    margin-top: 10px !important;
    margin-bottom: 0 !important;
}
.dayan-action-marker + div .stButton > button {
    min-height: 42px !important;
}
.dayan-action-marker + div [data-testid="column"] {
    min-height: 52px !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6b5320 0%, #8b6914 50%, #6b5320 100%) !important;
    border: 1px solid rgba(201,168,76,0.4) !important;
    color: #f0e8d0 !important;
    font-family: "Noto Serif TC", serif !important;
    letter-spacing: 3px !important;
    border-radius: 8px !important;
    transition: box-shadow 0.2s, transform 0.15s !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 16px rgba(201,168,76,0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    font-family: "Noto Serif TC", serif !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
}
hr {
    border-color: rgba(201,168,76,0.12) !important;
}
.sidebar-section {
    font-size: 14px;
    font-weight: bold;
    color: #FF4B4B;
    margin-top: 8px;
    margin-bottom: 4px;
}
.mode-badge-auto {
    background-color: #1f6aa5;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: bold;
}
.mode-badge-manual {
    background-color: #7a3b8c;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: bold;
}
.mode-badge-dayan {
    background-color: #8b6914;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

tab_dayan, tab_classic, tab_jiegua, tab_najia, tab_tools = st.tabs(
    [" 🌾排盤 ", " 🧮經典排盤 ", " 🚀解卦 ", " ☯️納甲 ", " 🔧工具 "]
)

# ---------------------------------------------------------------------------
# Sidebar – date/time, manual yao, and AI settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("#### 📅 起卦時間")
    pp_date = st.date_input("日期", pdlm.now(tz='Asia/Shanghai').date())

    if 'pp_time' not in st.session_state:
        st.session_state.pp_time = pdlm.now(tz='Asia/Shanghai').time()

    time_col, now_col = st.columns([3, 1])
    with time_col:
        pp_time = st.time_input("時間", value=st.session_state.pp_time)
        st.session_state.pp_time = pp_time
    with now_col:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("現在", help="設定為當前時間"):
            st.session_state.pp_time = pdlm.now(tz='Asia/Shanghai').time()
            st.rerun()

    p = str(pp_date).split("-")
    pp = str(pp_time).split(":")
    y = int(p[0])
    m = int(p[1])
    d = int(p[2])
    h = int(pp[0])
    mi = int(pp[1])

    st.divider()
    st.markdown("#### ✍️ 手動起爻")
    st.caption("初爻由下而上")

    with st.form("manual_form"):
        option_sixth = st.selectbox('上爻', ('老陰', '少陰', '少陽', '老陽'))
        option_fifth = st.selectbox('五爻', ('老陰', '少陰', '少陽', '老陽'))
        option_forth = st.selectbox('四爻', ('老陰', '少陰', '少陽', '老陽'))
        option_third = st.selectbox('三爻', ('老陰', '少陰', '少陽', '老陽'))
        option_second = st.selectbox('二爻', ('老陰', '少陰', '少陽', '老陽'))
        option_first = st.selectbox('初爻', ('老陰', '少陰', '少陽', '老陽'))
        yaodict = {"老陰": "6", '少陽': "7", "老陽": "9", '少陰': "8"}
        combine = "".join([yaodict.get(i) for i in [option_first, option_second, option_third, option_forth, option_fifth, option_sixth]])
        manual = st.form_submit_button('🔮 手動起卦', use_container_width=True)

    # --- AI settings ---
    st.markdown("---")
    st.header("AI設置")

    ai_provider = st.selectbox(
        "AI 服務",
        options=["Cerebras", "自定義（OpenAI相容）"],
        key="ai_provider_selector",
    )

    if ai_provider == "Cerebras":
        st.selectbox(
            "AI 模型",
            options=CEREBRAS_MODEL_OPTIONS,
            index=0,
            key="cerebras_model_selector",
            help="\n".join(f"• {k}: {v}" for k, v in CEREBRAS_MODEL_DESCRIPTIONS.items()),
        )
        st.text_input(
            "Cerebras API Key（可選）",
            type="password",
            key="cerebras_api_key_input",
            help="可留空，留空時將使用 .streamlit/secrets.toml 或環境變量 CEREBRAS_API_KEY。",
        )
    else:
        st.text_input(
            "模型名稱",
            value=st.session_state.get("custom_ai_model_input", DEFAULT_CUSTOM_MODEL),
            key="custom_ai_model_input",
            help="例如：gpt-4o、claude-3-7-sonnet、gemini-2.5-pro 等。",
        )
        st.text_input(
            "API Key",
            type="password",
            key="custom_ai_api_key_input",
            help="輸入你使用的 AI 服務 API Key。",
        )
        st.text_input(
            "Server URL",
            value=st.session_state.get("custom_ai_server_input", DEFAULT_CUSTOM_SERVER),
            key="custom_ai_server_input",
            help="OpenAI 相容接口地址，例如 https://api.openai.com/v1。",
        )

    system_prompts_data = load_system_prompts()
    prompts_list = system_prompts_data.get("prompts", [])
    prompt_names = [pr["name"] for pr in prompts_list]
    selected_prompt = system_prompts_data.get("selected")

    if prompt_names:
        selected_index = 0
        if selected_prompt in prompt_names:
            selected_index = prompt_names.index(selected_prompt)

        selected_name = st.selectbox(
            "選擇系統提示",
            options=prompt_names,
            index=selected_index,
            key="system_prompt_selector",
            help="選擇用於AI模型的系統提示，指導其分析六爻排盤結果",
        )

        system_prompts_data["selected"] = selected_name

        selected_content = ""
        for pr in prompts_list:
            if pr["name"] == selected_name:
                selected_content = pr["content"]
                break

        if "system_prompt" not in st.session_state:
            st.session_state.system_prompt = selected_content
        elif selected_name != st.session_state.get("last_selected_prompt"):
            st.session_state.system_prompt = selected_content

        st.session_state.last_selected_prompt = selected_name

        new_content = st.text_area(
            "編輯系統提示",
            value=st.session_state.system_prompt,
            height=150,
            placeholder="範例：你是一位六爻大師，根據排盤數據提供詳細分析...",
            key="system_prompt_editor",
        )
        st.session_state.system_prompt = new_content

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 更新提示", key="update_prompt_button"):
                for pr in prompts_list:
                    if pr["name"] == selected_name:
                        pr["content"] = new_content
                        break
                if save_system_prompts(system_prompts_data):
                    st.toast(f"✅ 已更新系統提示 '{selected_name}'！")
        with col2:
            if st.button(
                "❌ 刪除提示",
                key="delete_prompt_button",
                disabled=len(prompts_list) <= 1,
            ):
                prompts_list = [
                    pr for pr in prompts_list if pr["name"] != selected_name
                ]
                system_prompts_data["prompts"] = prompts_list
                if selected_name == selected_prompt and prompts_list:
                    system_prompts_data["selected"] = prompts_list[0]["name"]
                if save_system_prompts(system_prompts_data):
                    st.toast(f"✅ 已刪除系統提示 '{selected_name}'！")
                    st.rerun()

    if "form_key_suffix" not in st.session_state:
        st.session_state.form_key_suffix = 0

    name_key = f"new_prompt_name_{st.session_state.form_key_suffix}"
    content_key = f"new_prompt_content_{st.session_state.form_key_suffix}"

    with st.expander("➕ 新增提示", expanded=False):
        new_prompt_name = st.text_input("新提示名稱", key=name_key)
        new_prompt_content = st.text_area(
            "新提示內容",
            height=100,
            placeholder="輸入AI分析指令...",
            key=content_key,
        )
        if st.button(
            "➕ 新增提示",
            key="add_prompt_button",
            disabled=not new_prompt_name or not new_prompt_content,
        ):
            if new_prompt_name in prompt_names:
                st.error(f"提示名稱 '{new_prompt_name}' 已存在。")
            else:
                prompts_list.append(
                    {"name": new_prompt_name, "content": new_prompt_content}
                )
                system_prompts_data["prompts"] = prompts_list
                if save_system_prompts(system_prompts_data):
                    st.session_state.form_key_suffix += 1
                    st.toast(f"✅ 已新增系統提示 '{new_prompt_name}'！")
                    st.rerun()

    if st.toggle("🔧 高級設置", key="advanced_settings_toggle"):
        st.session_state.ai_max_tokens = st.slider(
            "最大生成 Tokens",
            40000,
            DEFAULT_MAX_TOKENS,
            st.session_state.get("ai_max_tokens", DEFAULT_MAX_TOKENS),
            key="ai_max_tokens_slider",
            help="控制AI回應的最大長度",
        )
        st.session_state.ai_temperature = st.slider(
            "溫度 (專注 vs. 創意)",
            0.0,
            1.5,
            st.session_state.get("ai_temperature", DEFAULT_TEMPERATURE),
            step=0.05,
            key="ai_temperature_slider",
            help="較低值 (如 0.2) 更確定性；較高值 (如 0.8) 更隨機",
        )

# ---------------------------------------------------------------------------
# Tab: 排盤（沉浸式大衍筮法）
# ---------------------------------------------------------------------------

with tab_dayan:
    render_dayan_tab()

# ---------------------------------------------------------------------------
# Tab: 經典排盤
# ---------------------------------------------------------------------------

with tab_classic:
    if "classic_mode" not in st.session_state:
        st.session_state.classic_mode = "time"
    if "classic_combine" not in st.session_state:
        st.session_state.classic_combine = ""
    if "random_gua" not in st.session_state:
        st.session_state.random_gua = ""

    header_col, mode_col = st.columns([4, 1])
    with header_col:
        st.header('☯️ 堅六爻　經典排盤')
    with mode_col:
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        if manual:
            st.markdown('<span class="mode-badge-manual">✍️ 手動盤</span>', unsafe_allow_html=True)
        elif st.session_state.classic_mode == "random":
            st.markdown('<span class="mode-badge-dayan">🎲 大衍盤</span>', unsafe_allow_html=True)
        elif st.session_state.classic_mode == "dayan_manual":
            st.markdown('<span class="mode-badge-dayan">🌾 手動大衍</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="mode-badge-auto">🤖 時間盤</span>', unsafe_allow_html=True)

    st.caption(XI_CI_QUOTE)

    mode = st.radio(
        "起卦方式",
        options=["時間起卦", "隨機大衍筮法", "手動輸入（側邊欄）", "手動大衍結果"],
        horizontal=True,
        index={
            "time": 0,
            "random": 1,
            "manual": 2,
            "dayan_manual": 3,
        }.get(
            "manual" if manual else st.session_state.classic_mode,
            0,
        ),
        key="classic_mode_radio",
    )

    mode_map = {
        "時間起卦": "time",
        "隨機大衍筮法": "random",
        "手動輸入（側邊欄）": "manual",
        "手動大衍結果": "dayan_manual",
    }
    st.session_state.classic_mode = mode_map[mode]

    if mode == "隨機大衍筮法":
        if st.button("🎲 隨機大衍起卦", type="primary"):
            st.session_state.random_gua = ichingshifa.Iching().bookgua()
            st.rerun()
        if st.session_state.random_gua:
            details = ichingshifa.Iching().mget_bookgua_details(st.session_state.random_gua)
            st.info(
                f"得卦：{st.session_state.random_gua}　"
                f"【{details[1]}】之【{details[2]}】"
            )

    if mode == "手動大衍結果":
        if st.session_state.dayan_combine:
            st.info(f"來自沉浸式排盤：{st.session_state.dayan_combine}")
        else:
            st.warning("請先在「排盤」tab 完成六爻起卦。")

    pan_text = ""
    output2 = st.empty()
    should_pan = (
        mode == "時間起卦"
        or (mode == "隨機大衍筮法" and st.session_state.random_gua)
        or (mode == "手動輸入（側邊欄）" and manual)
        or (mode == "手動大衍結果" and st.session_state.dayan_combine)
    )

    if should_pan:
        with st.spinner("起卦中，請稍候…"):
            with st_capture(output2.code):
                try:
                    if mode == "時間起卦":
                        pan_text = ichingshifa.Iching().display_pan(y, m, d, h, mi)
                        print(pan_text)
                    elif mode == "隨機大衍筮法":
                        pan_text = ichingshifa.Iching().display_pan_m(
                            y, m, d, h, mi, st.session_state.random_gua
                        )
                        print(pan_text)
                    elif mode == "手動輸入（側邊欄）":
                        pan_text = ichingshifa.Iching().display_pan_m(
                            y, m, d, h, mi, combine
                        )
                        print(pan_text)
                    elif mode == "手動大衍結果":
                        pan_text = ichingshifa.Iching().display_pan_m(
                            y, m, d, h, mi, st.session_state.dayan_combine
                        )
                        print(pan_text)
                except (ValueError, UnboundLocalError) as exc:
                    print(f"起卦錯誤：{exc}")

    if pan_text and st.button("🔍 使用AI分析排盤結果", key="analyze_with_ai"):
        with st.spinner("AI正在分析六爻排盤結果..."):
            try:
                user_prompt = (
                    "以下是六爻排盤的計算結果，請根據這些數據提供詳細的分析和解釋：\n\n"
                    + pan_text
                )
                messages = [
                    {
                        "role": "system",
                        "content": st.session_state.get("system_prompt", ""),
                    },
                    {"role": "user", "content": user_prompt},
                ]
                raw_response = request_ai_response(
                    messages=messages,
                    max_tokens=st.session_state.get("ai_max_tokens", DEFAULT_MAX_TOKENS),
                    temperature=st.session_state.get(
                        "ai_temperature", DEFAULT_TEMPERATURE
                    ),
                )
                with st.expander("AI分析結果", expanded=True):
                    st.markdown(raw_response)
            except Exception as e:
                st.error(f"調用AI時發生錯誤：{e}")

# ---------------------------------------------------------------------------
# Tab: 解卦
# ---------------------------------------------------------------------------

with tab_jiegua:
    st.header('解卦')
    sub_jue, sub_li = st.tabs(["占訣", "古占例"])
    with sub_jue:
        st.markdown(read_local_file("docs/text.md"))
    with sub_li:
        st.markdown(read_local_file("docs/example.md"))

# ---------------------------------------------------------------------------
# Tab: 納甲
# ---------------------------------------------------------------------------

with tab_najia:
    st.header('納甲')
    st.markdown(
        "漢元帝時期，易學名家**京房**開創京氏易學，將筮法融入干支納甲體系："
        "以八宮卦為綱，每爻配以天干地支，再結合五行生剋、六親"
        "（父母、官鬼、妻財、兄弟、子孫）、六獸"
        "（青龍、朱雀、勾陳、螣蛇、白虎、玄武）、五星、月建、日辰及二十八宿，"
        "構成完整的六爻象數推演體系。"
    )
    st.markdown(
        "> 「易有聖人之道四焉：以言者尚其辭，以動者尚其變，以制器者尚其象，以卜筮者尚其占。」\n"
        "> ——《繫辭傳》"
    )
    st.markdown("---")
    st.subheader("納甲排盤要素")
    st.markdown(
        """
| 要素 | 說明 |
|------|------|
| **世應** | 世爻代表占問方，應爻代表對方或事態 |
| **六親** | 父母、兄弟、子孫、妻財、官鬼 — 以日干五行為「我」推演 |
| **納甲** | 每爻所納天干地支，如「甲子」「庚寅」 |
| **六獸** | 青龍、朱雀、勾陳、螣蛇、白虎、玄武 |
| **伏神** | 本卦不見之六親，伏藏於他爻之下 |
| **旬空** | 日空、時空 — 該地支力量減弱 |
| **十二長生** | 長生、沐浴、冠帶、臨官、帝旺、衰、病、死、墓、絕、胎、養 |
        """
    )
    st.info(
        "完整納甲排盤請在「排盤」或「經典排盤」tab 起卦後查看。"
        "排盤結果包含本卦、之卦、互卦的完整納甲、六親、六獸、世應與伏神。"
    )

# ---------------------------------------------------------------------------
# Tab: 工具
# ---------------------------------------------------------------------------

with tab_tools:
    st.header('工具')
    sub_log, sub_link = st.tabs(["日誌", "連結"])
    with sub_log:
        st.markdown(read_local_file("docs/update.md"))
    with sub_link:
        try:
            st.markdown(
                get_remote_file(
                    "https://raw.githubusercontent.com/kentang2017/kinliuren/master/update.md"
                ),
                unsafe_allow_html=True,
            )
        except Exception:
            st.info("無法載入遠端連結內容。")

# ---------------------------------------------------------------------------
# Fixed bottom LLM chat
# ---------------------------------------------------------------------------

st.divider()
st.subheader("💬 AI 聊天")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_chat_input := st.chat_input("輸入您的問題…"):
    st.session_state.chat_messages.append({"role": "user", "content": user_chat_input})
    with st.chat_message("user"):
        st.markdown(user_chat_input)

    try:
        system_prompt_content = st.session_state.get("system_prompt", "")
        messages_payload = []
        if system_prompt_content:
            messages_payload.append({"role": "system", "content": system_prompt_content})
        messages_payload += [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
        ]
        with st.chat_message("assistant"):
            with st.spinner("AI 正在回應…"):
                assistant_reply = request_ai_response(
                    messages=messages_payload,
                    max_tokens=st.session_state.get("ai_max_tokens", DEFAULT_MAX_TOKENS),
                    temperature=st.session_state.get("ai_temperature", DEFAULT_TEMPERATURE),
                )
                st.markdown(assistant_reply)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": assistant_reply}
                )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"調用AI時發生錯誤：{e}")