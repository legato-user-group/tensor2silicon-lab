#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2S Lab WebBook (VitePress) Content Builder — Method A (방식 A)
Integrates:
1. Main Lecture Notes (/home/leeyb/t2s/[N주차 교안]*.md & tensor2silicon-lab-main/week0N/yblee.md)
2. Weekly Guides & Reference Calculation Examples (/home/leeyb/t2s/[N주차 안내]*.md & [N주차 참고자료]*.md)
3. Researcher Deep-Dive Highlights & Full Researcher Notes (week02 ~ week07)
4. Interactive Simulators & HTML Decks
"""

import os
import re
import glob
from pathlib import Path

ROOT_DIR = Path("/home/leeyb/t2s")
REPO_DIR = ROOT_DIR / "tensor2silicon-lab-main"
BOOK_DIR = ROOT_DIR / "t2s_gitbook"
DOCS_DIR = BOOK_DIR / "docs"

WEEKS_META = {
    2: {
        "title": "모델에서 실리콘까지",
        "subtitle": "AI 실행 스택 5대 계층 해부 — PyTorch, JAX·OpenXLA, Modular MAX·Mojo, Triton",
        "question": "모델의 연산 하나가 가속기에서 실행되기까지 어떤 단계를 거치며, 각 단계는 무엇을 담당할까?",
        "main_src": REPO_DIR / "week02" / "yblee.md",
        "guide_src": ROOT_DIR / "[2주차 안내] 모델에서 실리콘까지.md",
        "ref_src": None,
        "researchers": [
            {
                "id": "mbsong",
                "name": "mbsong (송민배)",
                "badge": "PyTorch·XLA·MAX 2부작",
                "files": [REPO_DIR / "week02" / "mbsong-1.md", REPO_DIR / "week02" / "mbsong-2.md"],
                "summary": "PyTorch Eager/Compile부터 JAX·OpenXLA, Modular MAX·Mojo까지 실행 스택의 IR 변환과 런타임 디스패치 과정을 2부작으로 정밀 비교했습니다."
            },
            {
                "id": "singon",
                "name": "singon (김신곤)",
                "badge": "다이어그램 & 스택 해부",
                "files": [REPO_DIR / "week02" / "singon.md"],
                "summary": "풍부한 아키텍처 다이어그램을 바탕으로 모델 수식이 그래프 캡처(Dynamo/FX)와 컴파일러를 거쳐 실리콘 명령어로 내려가는 흐름을 시각화했습니다."
            },
            {
                "id": "junpark",
                "name": "junpark (박준)",
                "badge": "실행 스택 핵심 요약",
                "files": [REPO_DIR / "week02" / "junpark.md"],
                "summary": "Python 바이트코드 변환(TorchDynamo)과 GPU 하드웨어 계층 구조의 연결 고리를 핵심 도식 중심으로 압축 정리했습니다."
            },
            {
                "id": "injokim",
                "name": "injokim (김인조)",
                "badge": "인터랙티브 HTML & 슬라이드",
                "files": [],
                "html_decks": [
                    {
                        "src": "/decks/injokim-week02.html",
                        "title": "AI 실행 스택 인터랙티브 학습 노트 (injokim.html)",
                        "desc": "김인조 연구원이 제작한 인터랙티브 실행 스택 비교 및 학습 노트입니다."
                    },
                    {
                        "src": "/decks/torch-compile-anatomy-deck.html",
                        "title": "torch.compile 내부 구조 해부 슬라이드 덱",
                        "desc": "TorchDynamo → AOTAutograd → PrimTorch → TorchInductor → Triton Codegen 전 과정을 담은 인터랙티브 프레젠테이션입니다."
                    }
                ],
                "summary": "정적 문서를 넘어 브라우저에서 직접 클릭하며 탐색할 수 있는 인터랙티브 학습 노트(HTML)와 torch.compile 해부 슬라이드 덱을 제작했습니다."
            },
            {
                "id": "jaehyeong",
                "name": "jaehyeong (이재형)",
                "badge": "컴파일 파이프라인 정리",
                "files": [REPO_DIR / "week02" / "jaehyeong.md"],
                "summary": "프레임워크별 그래프 캡처 방식, IR 계층의 존재 이유, 커널 퓨전(Fusion)이 메모리 왕복을 줄이는 원리를 체계적으로 정리했습니다."
            }
        ]
    },
    3: {
        "title": "커널의 시간은 어디에서 소비되는가",
        "subtitle": "연산 시간 vs 메모리 이동 시간, HBM 대역폭, Vector Add와 Kernel Fusion",
        "question": "같은 연산인데 왜 어떤 커널은 빠르고 어떤 커널은 느릴까? 계산 시간과 데이터 이동 시간을 어떻게 구분할까?",
        "main_src": ROOT_DIR / "[3주차 교안] 커널의 시간은 어디에서 소비되는가.md",
        "guide_src": ROOT_DIR / "[3주차 안내] 커널의 시간은 어디에서 소비되는가.md",
        "ref_src": ROOT_DIR / "[3주차 참고자료] 계산 예제와 연습문제.md",
        "researchers": [
            {
                "id": "mbsong",
                "name": "mbsong (송민배)",
                "badge": "GPU·TPU 메모리 계층 비교",
                "files": [REPO_DIR / "week03" / "mbsong.md"],
                "summary": "Blackwell SM 구조와 TPU TensorCore(MXU/VPU/VMEM) 아키텍처를 대조하며 연산 시간·메모리 이동 시간·런치 오버헤드를 정량 분석했습니다."
            },
            {
                "id": "singon",
                "name": "singon (김신곤)",
                "badge": "손계산 & 병목 판별",
                "files": [REPO_DIR / "week03" / "singon.md"],
                "summary": "Vector Add와 다단계 원소별 연산에서 HBM 읽기/쓰기 바이트 수를 직접 손계산하고 Kernel Fusion의 시간 절감 효과를 도출했습니다."
            },
            {
                "id": "junpark",
                "name": "junpark (박준)",
                "badge": "SRAM vs DRAM 물리 구조",
                "files": [REPO_DIR / "week03" / "junpark.md"],
                "summary": "왜 온칩 SRAM은 빠르고 작으며 오프칩 DRAM(HBM)은 크고 느린지 6T SRAM 셀과 1T1C DRAM 뱅크의 물리적 회로 구조부터 명쾌하게 설명했습니다."
            },
            {
                "id": "injokim",
                "name": "injokim (김인조)",
                "badge": "수식 모델링 & 정량 분석",
                "files": [REPO_DIR / "week03" / "injokim.md"],
                "summary": "커널 실행 시간 분해 모델(Launch Overhead, Compute Time, Memory Time)과 실효 대역폭(Effective Bandwidth) 수식을 체계적으로 정립했습니다."
            },
            {
                "id": "jaehyeong",
                "name": "jaehyeong (이재형)",
                "badge": "메모리 계층 & 실전 예제",
                "files": [REPO_DIR / "week03" / "jaehyeong.md"],
                "summary": "GPU 메모리 공간(Global, L2, Shared, Register)의 특성과 Activation 레이어가 Memory-Bound에 걸리는 실제 사례를 분석했습니다."
            }
        ]
    },
    4: {
        "title": "Roofline과 Tiling",
        "subtitle": "산술 강도(I = F/Q), Roofline 상한 모델, 온칩 메모리 데이터 재사용",
        "question": "같은 계산을 하면서 데이터를 덜 읽어오면 커널은 얼마나 빨라질 수 있을까?",
        "main_src": ROOT_DIR / "[4주차 교안] Roofline과 Tiling.md",
        "guide_src": ROOT_DIR / "[4주차 안내] Roofline과 Tiling.md",
        "ref_src": ROOT_DIR / "[4주차 참고자료] Roofline과 Tiling 계산 예제와 연습문제.md",
        "researchers": [
            {
                "id": "mbsong",
                "name": "mbsong (송민배)",
                "badge": "Roofline & 타일 크기별 분석",
                "files": [REPO_DIR / "week04" / "mbsong.md"],
                "summary": "타일 크기(t=32, 64, 128)에 따른 HBM 이동량 감소와 산술 강도 변화를 로그 스케일 Roofline 그래프와 함께 완벽히 유도했습니다."
            },
            {
                "id": "singon",
                "name": "singon (김신곤)",
                "badge": "4×4 & 4096×4096 완전 해설",
                "files": [REPO_DIR / "week04" / "singon.md"],
                "summary": "작은 4×4 행렬곱 타일링의 손가락 추적부터 4096×4096 대규모 행렬곱의 온칩 버퍼 용량 계산까지 전 과정을 상세히 풀어냈습니다."
            },
            {
                "id": "junpark",
                "name": "junpark (박준)",
                "badge": "2D Block Tiling 직관",
                "files": [REPO_DIR / "week04" / "junpark.md"],
                "summary": "Nsight Compute의 Roofline 시각화와 CUDA 2D Block Tiling 다이어그램을 연결해 타일 재사용의 기하학적 의미를 정리했습니다."
            },
            {
                "id": "injokim",
                "name": "injokim (김인조)",
                "badge": "이론 유도 & 연습문제 풀이",
                "files": [REPO_DIR / "week04" / "injokim.md"],
                "summary": "산술 강도 정의, 하드웨어 분기점(Ridge Point) 이동 조건, 타일 크기 증가에 따른 온칩 SRAM 제약의 트레이드오프를 수식으로 정리했습니다."
            },
            {
                "id": "jaehyeong",
                "name": "jaehyeong (이재형)",
                "badge": "Roofline & Tiling 핵심 노트",
                "files": [REPO_DIR / "week04" / "jaehyeong.md"],
                "summary": "Compute-bound와 Bandwidth-bound의 경계 조건 및 타일 행렬곱에서 입력 조각이 재사용되는 횟수를 직관적으로 요약했습니다."
            }
        ]
    },
    5: {
        "title": "xPU를 어떻게 비교할 것인가",
        "subtitle": "GPU, TPU, NPU/LPU 아키텍처 비교 기준 — 연산 유닛, 메모리 계층, 제어 모델",
        "question": "GPU, TPU, NPU/LPU 등 서로 다른 가속기 아키텍처와 프로그래밍 모델을 어떤 기준으로 비교할까?",
        "main_src": ROOT_DIR / "[5주차 교안] xPU를 어떻게 비교할 것인가.md",
        "guide_src": ROOT_DIR / "[5주차 안내] xPU를 어떻게 비교할 것인가.md",
        "ref_src": ROOT_DIR / "[5주차 참고자료] xPU 비교 기준과 계산 예제.md",
        "researchers": [
            {
                "id": "mbsong",
                "name": "mbsong (송민배)",
                "badge": "GPU vs TPU vs NPU 비교표",
                "files": [REPO_DIR / "week05" / "mbsong.md"],
                "summary": "SIMT 기반 GPU(H100)와 시스톨릭 어레이 기반 TPU(v5e/v6e)의 실행 모델, 캐시 vs 스크래치패드(VMEM) 제어 방식을 체계적으로 대조했습니다."
            },
            {
                "id": "singon",
                "name": "singon (김신곤)",
                "badge": "xPU 비교 기준 & 손계산",
                "files": [REPO_DIR / "week05" / "singon.md"],
                "summary": "스펙시트의 피크 TFLOPs 숫자에 속지 않고 실제 워크로드(Prefill vs Decode)와 산술 강도로 xPU를 공정하게 비교하는 방법을 정리했습니다."
            },
            {
                "id": "junpark",
                "name": "junpark (박준)",
                "badge": "아키텍처 심층 리서치",
                "files": [REPO_DIR / "week05" / "junpark.md"],
                "summary": "GPU 워프 스케줄러, TPU MXU 시스톨릭 데이터 흐름, LPU/NPU의 결정론적 실행 구조를 하드웨어 마이크로아키텍처 관점에서 분석했습니다."
            }
        ]
    },
    6: {
        "title": "Python에서 Triton 커널로",
        "subtitle": "1D 블록 프로그래밍(tl.arange, mask), Fused Online Softmax, 온칩 SRAM 재사용",
        "question": "Python 기반 커널 DSL(Triton)로 블록 단위 프로그래밍과 메모리 접근을 어떻게 제어할까?",
        "main_src": ROOT_DIR / "[6주차 교안] Python에서 Triton 커널로.md",
        "guide_src": ROOT_DIR / "[6주차 안내] Python에서 Triton 커널로.md",
        "ref_src": None,
        "researchers": [
            {
                "id": "mbsong",
                "name": "mbsong (송민배)",
                "badge": "Vector Add & Fused Softmax",
                "files": [REPO_DIR / "week06" / "mbsong.md"],
                "summary": "CUDA 스레드 인덱싱과 Triton 블록 인덱싱을 대조하고, Fused Softmax가 HBM 왕복을 1/4로 줄이는 원리를 코드와 수식으로 정리했습니다."
            },
            {
                "id": "singon",
                "name": "singon (김신곤)",
                "badge": "RTX 4070 실측 & 벤치마크",
                "files": [REPO_DIR / "week06" / "singon.md"],
                "summary": "RTX 4070에서 Naive Softmax, PyTorch 내장 Softmax, Triton Fused Softmax의 실행 시간과 실효 대역폭을 직접 측정·검증했습니다."
            },
            {
                "id": "junpark",
                "name": "junpark (박준)",
                "badge": "Online Softmax & IR 분석",
                "files": [REPO_DIR / "week06" / "junpark.md"],
                "summary": "Online Softmax의 수치 안정성 보정 수식과 Triton 커널이 워프 리덕션(Reduction) 명령으로 컴파일되는 과정을 깊게 파헤쳤습니다."
            },
            {
                "id": "kiminjo",
                "name": "injokim (김인조)",
                "badge": "블록 포인터 & 마스킹 정밀 해부",
                "files": [REPO_DIR / "week06" / "kiminjo.md"],
                "summary": "1D 포인터 산술, next_power_of_2 패딩, 마스킹 조건(other=-inf vs 0.0)이 결과 정확도와 성능에 미치는 영향을 상세히 분석했습니다."
            },
            {
                "id": "jaehyeong",
                "name": "jaehyeong (이재형)",
                "badge": "Triton 프로그래밍 핵심 요약",
                "files": [REPO_DIR / "week06" / "Jaehyeong.md"],
                "summary": "SPMD 프로그램 인스턴스(program_id) 개념과 블록 단위 로드/스토어, 커널 퓨전의 핵심 패턴을 직관적으로 요약했습니다."
            }
        ]
    },
    7: {
        "title": "Tiled MatMul과 컴파일 과정",
        "subtitle": "2D 포인터 브로드캐스팅, L2 Grouped Ordering, Autotune, 다단계 IR(TTIR→TTGIR→PTX→SASS)",
        "question": "Python으로 작성한 2D Tiled MatMul 커널은 어떻게 데이터를 재사용하며, 어떤 컴파일 단계를 거쳐 GPU 기계어가 될까?",
        "main_src": ROOT_DIR / "[7주차 교안] Tiled MatMul과 컴파일 과정.md",
        "guide_src": ROOT_DIR / "[7주차 안내] Tiled MatMul과 컴파일 과정.md",
        "ref_src": ROOT_DIR / "[7주차 참고자료] Tiled MatMul과 컴파일 과정 계산 예제와 연습문제.md",
        "researchers": [
            {
                "id": "mbsong",
                "name": "mbsong (송민배)",
                "badge": "NVIDIA A100 IR·SASS 실측",
                "files": [REPO_DIR / "week07" / "mbsong.md"],
                "summary": "A100(sm_80)에서 추출한 TTIR·TTGIR·LLVM IR·PTX·SASS 원본 산출물을 바탕으로 레이아웃 변환과 cp.async/mma.sync 매핑을 실증했습니다."
            },
            {
                "id": "junpark",
                "name": "junpark (박준)",
                "badge": "RTX 5090 (sm_120) 실측 해부",
                "files": [REPO_DIR / "week07" / "junpark.md"],
                "summary": "최신 Blackwell RTX 5090(sm_120)에서 mma.sync.aligned.m16n8k16 명령 개수(워프당 64개)와 num_stages=4 순환 버퍼 동작을 PTX 수준에서 검증했습니다."
            },
            {
                "id": "singon",
                "name": "singon (김신곤)",
                "badge": "L2 그룹핑 손계산 & IR 대응표",
                "files": [REPO_DIR / "week07" / "singon.md"],
                "summary": "Row-major 순서와 Grouped Ordering(GROUP_SIZE_M)의 L2 캐시 적중 차이를 손계산으로 증명하고, Python 코드와 각 IR 단계를 1:1로 대응시켰습니다."
            },
            {
                "id": "injokim",
                "name": "injokim (김인조)",
                "badge": "알고리즘 & MLIR 하향 변환 종합",
                "files": [REPO_DIR / "week07" / "injokim.md"],
                "summary": "2D 브로드캐스팅 주소 수식 유도부터 MLIR Dialect(tt → ttg → llvm) 하향 변환 파이프라인까지 학술 보고서 수준으로 체계화했습니다."
            },
            {
                "id": "jaehyeong",
                "name": "jaehyeong (이재형)",
                "badge": "핵심 코드 & 컴파일 단계 요약",
                "files": [REPO_DIR / "week07" / "jaehyeong.md"],
                "summary": "프로그램 ID의 2차원 타일 좌표 매핑, K축 포인터 전진, 오토튜닝 메타파라미터와 다단계 IR의 핵심 역할을 깔끔하게 정리했습니다."
            }
        ]
    }
}


def sanitize_markdown_for_vitepress(md_text: str, week_num: int, is_researcher: bool = False) -> str:
    """
    Cleans up markdown edge cases so VitePress (Vue 3 + markdown-it-mathjax3 + Mermaid) compiles cleanly:
    1. Fixes malformed ![alt](<img ... src="URL" />) in week06/singon.md
    2. Maps /home/leeyb/t2s/assets/week04/... and ./assets/roofline.png to /assets/week04/...
    3. Maps mbsong's relative repo links/images to Jake-Song/tensor2silicon GitHub URLs
    4. Replaces missing Notion-exported local image paths (e.g. image.png, image%201.png) with a clean notice badge
    5. Protects raw <angle_brackets> and {{ curly }} outside of code blocks/math from breaking Vue template compiler
    """
    # 1. Fix malformed ![alt](<img ... src="URL" />
    md_text = re.sub(
        r'!\[([^\]]*)\]\(<img[^>]*src="([^"]+)"[^>]*\/?>\)?',
        r'![\1](\2)',
        md_text
    )

    # 2. Map local /home/leeyb/t2s/assets/... to VitePress public path (relative to site root or relative path)
    # Note: In VitePress, paths starting with /assets/... in markdown are resolved from docs/public/assets/...
    md_text = md_text.replace("/home/leeyb/t2s/assets/week04/", "/assets/week04/")
    if week_num == 4:
        md_text = md_text.replace("(./assets/roofline.png)", "(/assets/week04/roofline.png)")

    # 3. Map mbsong's relative assets and links to his GitHub repo so they work on the web
    md_text = md_text.replace("(./assets/gpu-diagram.png)", "(https://jax-ml.github.io/scaling-book/assets/gpu/gpu-diagram.png)")
    md_text = md_text.replace("(./assets/blackwell-sm.png)", "(https://raw.githubusercontent.com/Jake-Song/tensor2silicon/main/week3/assets/blackwell-sm.png)")
    md_text = md_text.replace("(./assets/tpu-chip.png)", "(https://raw.githubusercontent.com/Jake-Song/tensor2silicon/main/week3/assets/tpu-chip.png)")
    md_text = md_text.replace("(./assets/pointwise-product.gif)", "(https://raw.githubusercontent.com/Jake-Song/tensor2silicon/main/week3/assets/pointwise-product.gif)")

    # Map mbsong's relative links in week07/week06/week05/week04/week03
    md_text = re.sub(
        r'\]\(\.\./week(\d+)/[^\)]*\)',
        lambda m: f'](/week0{m.group(1)}/)',
        md_text
    )
    md_text = re.sub(
        r'\]\(\.\./notebooks/([^\)]+)\)',
        r'](https://github.com/Jake-Song/tensor2silicon/blob/main/notebooks/\1)',
        md_text
    )
    md_text = re.sub(
        r'\]\((compiled_artifacts_summary\.md|artifacts/[^\)]+)\)',
        r'](https://github.com/Jake-Song/tensor2silicon/tree/main/week7/\1)',
        md_text
    )

    # 4. Handle missing Notion local images in jaehyeong.md (including nested link inside alt text like ![[url](url)](image%204.png))
    md_text = re.sub(
        r'!\[\[([^\]]+)\]\(([^)]+)\)\]\(([^)]+)\)',
        r'> 🖼️ *[도식 출처: [\1](\2)]*',
        md_text
    )

    def replace_missing_local_img(match):
        alt = match.group(1)
        url = match.group(2).strip()
        if url.startswith(("http://", "https://", "/assets/", "/decks/")):
            return match.group(0)
        # Missing local image from Notion export
        clean_alt = alt if alt and alt != "image.png" else "원본 노션 첨부 도식"
        return f'> 🖼️ *[도식 참고: {clean_alt}]*'

    md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_missing_local_img, md_text)

    # 5. Protect raw <...> and {{ ... }} outside fenced code blocks and inline code
    lines = md_text.split("\n")
    out_lines = []
    in_fence = False
    fence_marker = ""

    # Valid HTML tags we allow in markdown
    allowed_html_tags = {
        "div", "span", "p", "br", "hr", "img", "a", "b", "i", "strong", "em",
        "code", "pre", "blockquote", "ul", "ol", "li", "table", "thead", "tbody",
        "tr", "th", "td", "details", "summary", "sup", "sub", "kbd", "mark",
        "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "iframe",
        "WeekNavBar", "HtmlDeckViewer", "RooflineCalculator", "IrPipelineViewer"
    }

    for line in lines:
        stripped = line.lstrip()
        if not in_fence:
            m = re.match(r'^(`{3,}|~{3,})', stripped)
            if m:
                in_fence = True
                fence_marker = m.group(1)
                out_lines.append(line)
                continue
        else:
            if stripped.startswith(fence_marker) and re.match(r'^(`{3,}|~{3,})\s*$', stripped):
                in_fence = False
                fence_marker = ""
            out_lines.append(line)
            continue

        # Outside fenced code block: split by inline code (`...`) and math ($...$) so we only sanitize plain text
        # Wait: let's split by inline backticks `...` first
        parts = re.split(r'(`+[^`]+`+)', line)
        for idx in range(0, len(parts), 2):
            segment = parts[idx]
            # Escape {{ and }} if outside math, or even inside math if Vue complains?
            # Actually in MathJax3, `\frac{a}{b}` is fine, `\frac{{a}}{{b}}` has `{{` which Vue template compiler may try to interpolate if mathjax doesn't consume it or if it's in text!
            # Putting a space `{ {` in LaTeX is 100% identical in MathJax and completely prevents Vue interpolation!
            segment = segment.replace("{{", "{ {").replace("}}", "} }")

            # Also check for `<something>` that is not a valid HTML tag or HTML comment
            def fix_angle_brackets(m):
                full = m.group(0)
                inner = m.group(1)
                # Check if comment
                if inner.startswith("!--"):
                    return full
                # Extract tag name
                tag_match = re.match(r'^/?([a-zA-Z0-9_-]+)', inner.strip())
                if tag_match:
                    tag_name = tag_match.group(1)
                    if tag_name in allowed_html_tags or tag_name.lower() in allowed_html_tags:
                        return full
                # Otherwise escape < and >
                return "&lt;" + inner + "&gt;"

            segment = re.sub(r'<([^<>]+)>', fix_angle_brackets, segment)
            parts[idx] = segment

        out_lines.append("".join(parts))

    return "\n".join(out_lines)


def wrap_solutions_in_details(md_text: str) -> str:
    """
    Finds the '해답과 해설' or '연습문제 해답' section in a lecture note / reference file
    and wraps its body inside a collapsible <details class="t2s-solution-toggle"> block!
    """
    pattern = re.compile(
        r'^(##\s+(?:\d+\.\s*)?(?:해답과 해설|연습문제 해답[^\n]*|해답 및 해설[^\n]*))\n(.*?)(?=^##\s+|\Z)',
        re.MULTILINE | re.DOTALL
    )

    def repl(match):
        heading = match.group(1)
        body = match.group(2).strip()
        return (
            f"{heading}\n\n"
            f'<details class="t2s-solution-toggle">\n'
            f'<summary>💡 연습문제 해답 및 상세 풀이 펼치기 (스스로 먼저 풀어본 뒤 클릭하세요)</summary>\n\n'
            f"{body}\n\n"
            f"</details>\n\n"
        )

    return pattern.sub(repl, md_text)


def build_week_nav_bar_tag(week_num: int) -> str:
    meta = WEEKS_META[week_num]
    researchers_list = [
        {"id": r["id"], "name": r["name"].split(" ")[0], "label": r["name"], "badge": r["badge"]}
        for r in meta["researchers"]
    ]
    import json
    r_json = json.dumps(researchers_list, ensure_ascii=False).replace('"', "'")
    return f'<WeekNavBar :week="{week_num}" topic="{meta["title"]}" :researchers="{r_json}" />\n\n'


def build_researcher_synthesis_section(week_num: int) -> str:
    meta = WEEKS_META[week_num]
    cards = []
    for r in meta["researchers"]:
        cards.append(
            f'  <div class="t2s-synthesis-card">\n'
            f'    <div>\n'
            f'      <div class="t2s-synthesis-card-head">\n'
            f'        <span class="t2s-researcher-handle">🧑‍🔬 {r["name"]}</span>\n'
            f'        <span class="t2s-researcher-tag">{r["badge"]}</span>\n'
            f'      </div>\n'
            f'      <p>{r["summary"]}</p>\n'
            f'    </div>\n'
            f'    <a class="t2s-synthesis-link" href="./researchers/{r["id"]}">원문 노트 & 실측 분석 읽기 →</a>\n'
            f'  </div>'
        )
    cards_html = "\n".join(cards)
    return (
        f"\n## 🔬 이번 주 연구원 리서치 & 실측 하이라이트 (공동 집필 아카이브)\n\n"
        f"> 본 주차 주제를 랩 연구원들이 각자의 관점과 하드웨어 환경에서 교차 검증·심화 분석한 핵심 요약입니다. 카드를 클릭하면 각 연구원의 전체 리서치 노트로 이동합니다.\n\n"
        f'<div class="t2s-synthesis-grid">\n{cards_html}\n</div>\n\n---\n\n'
    )


def build_week_interactive_section(week_num: int) -> str:
    if week_num == 2:
        return (
            "\n## 🖥️ 인터랙티브 실행 스택 아키텍처 & 슬라이드 덱\n\n"
            "> 2주차 연구원(`injokim`)이 제작한 인터랙티브 학습 노트와 `torch.compile` 내부 구조 슬라이드 덱입니다. 페이지 안에서 바로 조작하거나 **전체화면**으로 열람할 수 있습니다.\n\n"
            '<HtmlDeckViewer src="/decks/injokim-week02.html" title="PyTorch · OpenXLA · Modular MAX 실행 스택 인터랙티브 노트" author="injokim (김인조)" :height="580" />\n\n'
            '<HtmlDeckViewer src="/decks/torch-compile-anatomy-deck.html" title="torch.compile 내부 구조 해부 (Dynamo → AOTAutograd → Inductor)" author="T2S Lab" :height="580" />\n\n---\n\n'
        )
    elif week_num == 3:
        return (
            "\n## 🎛️ 인터랙티브 실습: 연산 병목 vs 메모리 대역폭 병목 계산기\n\n"
            "> 하드웨어 대역폭(BW)과 피크 연산 성능(P), 그리고 데이터 재사용 여부에 따라 커널의 성능 상한이 어떻게 결정되는지 직접 조건을 바꿔가며 확인해 보세요.\n\n"
            "<RooflineCalculator />\n\n---\n\n"
        )
    elif week_num == 4:
        return (
            "\n## 🎛️ 인터랙티브 Roofline & Tiling 시뮬레이터\n\n"
            "> 4주차 공통 예제(`N = 4096`, `FP32`, `P = 20 TFLOPs/s`, `BW = 1 TB/s`)에서 타일 크기(`16×16` ~ `256×256`)를 바꿀 때 **HBM 메모리 이동량(Q)**과 **산술 강도(I = F/Q)**가 어떻게 변하며 언제 연산 병목 구간에 진입하는지 직접 실험해 보세요.\n\n"
            "<RooflineCalculator />\n\n"
            "### 📊 4주차 핵심 시각화 도표 모음\n\n"
            "| Roofline 성능 상한과 재사용 조건 | 4×4 행렬곱의 2×2 출력 타일 누적 과정 |\n"
            "| :---: | :---: |\n"
            "| ![Roofline 그래프](/assets/week04/roofline.svg) | ![Tiling 누적 과정](/assets/week04/tiling.svg) |\n\n"
            "<details class=\"t2s-solution-toggle\">\n"
            "<summary>🖼️ 추가 시각 자료 펼치기 (타일 저장 공간·BK 폭 비교·하드웨어 변화에 따른 Roofline 이동)</summary>\n\n"
            "![타일 저장 공간과 병렬성](/assets/week04/lecture-visuals/tile_storage_parallelism.svg)\n\n"
            "![BK 폭 비교](/assets/week04/lecture-visuals/bk_width_comparison.svg)\n\n"
            "![하드웨어 변화와 Roofline](/assets/week04/lecture-visuals/roofline_hardware_changes.svg)\n\n"
            "</details>\n\n---\n\n"
        )
    elif week_num == 6:
        return (
            "\n## 🔍 미리 보기: Triton 커널이 GPU 기계어가 되기까지\n\n"
            "> 6주차에서 작성한 Triton 블록 커널(`@triton.jit`)이 7주차에서 다루는 다단계 MLIR 파이프라인(`TTIR` → `TTGIR` → `LLVM IR` → `PTX` → `SASS`)을 거쳐 어떻게 변환되는지 단계별로 미리 살펴볼 수 있습니다.\n\n"
            "<IrPipelineViewer />\n\n---\n\n"
        )
    elif week_num == 7:
        return (
            "\n## 🎛️ 인터랙티브 실습 1: 2D Tiled MatMul & 다단계 IR 파이프라인 탐색기\n\n"
            "> 아래 두 가지 인터랙티브 도구를 통해 **① 타일 크기(`BM, BN, BK`)와 파이프라인 단계(`num_stages`)에 따른 Shared Memory 요구량 및 Roofline 성능 상한**을 계산하고, **② Python 코드가 TTIR → TTGIR → LLVM IR → PTX → SASS로 변환되는 과정(A100 vs RTX 5090 실측 비교)**을 단계별로 확인해 보세요.\n\n"
            "<IrPipelineViewer />\n\n"
            "<RooflineCalculator />\n\n---\n\n"
        )
    return ""


def generate_all():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "intro").mkdir(parents=True, exist_ok=True)

    # 1. Build Home Page (docs/index.md)
    index_md = """---
layout: home

hero:
  name: "Tensor2Silicon LAB"
  text: "AI 모델에서 가속기 커널까지"
  tagline: "PyTorch·Transformer 수식 한 줄이 그래프(IR), 컴파일러, 런타임을 거쳐 GPU·TPU·NPU 실리콘 위에서 실행되기까지의 전 과정을 정량적 손계산과 커널 실습으로 파헤치는 오픈 리서치 웹북"
  actions:
    - theme: brand
      text: "📖 웹북 읽기 시작하기 (랩 소개)"
      link: /intro/overview
    - theme: alt
      text: "🚀 2주차: 모델에서 실리콘까지"
      link: /week02/
    - theme: alt
      text: "🔥 7주차: Tiled MatMul & IR 해부"
      link: /week07/

features:
  - icon: 🧮
    title: "정량적 손계산 & 인터랙티브 Roofline"
    details: "막연한 개념 설명 대신 FLOPs, HBM 메모리 이동량(Q), 산술 강도(I = F/Q), Shared Memory 타일 버퍼 크기를 손으로 직접 계산하고 웹북 내장 시뮬레이터로 즉시 실험합니다."
  - icon: ⚙️
    title: "다단계 컴파일러 & IR 실측 해부"
    details: "PyTorch(torch.compile), JAX(OpenXLA), Modular MAX, 그리고 Triton 커널이 TTIR → TTGIR → LLVM IR → PTX → SASS로 구체화되는 과정을 A100 및 RTX 5090 실측 데이터로 검증합니다."
  - icon: 🧑‍🔬
    title: "방식 A: 메인 교안 + 6인 연구원 Deep-Dive"
    details: "체계적인 메인 교과서 흐름을 따라가면서도, 매 주차 6명의 연구원이 각자의 하드웨어와 시각에서 분석한 심화 리포트와 인터랙티브 슬라이드를 탭 한 번으로 함께 열람할 수 있습니다."
  - icon: 🌐
    title: "Legato 오픈소스 저수준 IR 생태계"
    details: "과학기술정보통신부·IITP K-클라우드 기술개발 사업의 일환으로, 국산 AI반도체 및 이기종 가속기를 위한 개방형 저수준 IR(Legato) 커뮤니티의 기술적 토대를 구축합니다."
---

## 🗺️ 커리큘럼 한눈에 바로가기 (Curriculum Roadmap)

| 주차 | 주제 (Topic) | 핵심 내용 및 인터랙티브 도구 | 바로가기 |
| :---: | :--- | :--- | :---: |
| **Intro** | **랩 소개 & 5대 실행 스택** | T2S Lab 운영 원칙, 5대 계층 아키텍처, 연구원 아카이브 매트릭스 | [소개 보기](./intro/overview) |
| **Week 02** | **모델에서 실리콘까지** | PyTorch 2.0, JAX·OpenXLA, Modular MAX·Mojo 비교 + **인터랙티브 HTML 슬라이드 덱** | [2주차 열기](./week02/) |
| **Week 03** | **커널의 시간은 어디에서 소비되는가** | 연산 시간 vs 메모리 이동 시간, SRAM vs DRAM 물리 구조, Vector Add & Fusion | [3주차 열기](./week03/) |
| **Week 04** | **Roofline과 Tiling** | 산술 강도($I = F/Q$), 타일 데이터 재사용, **인터랙티브 Roofline 시뮬레이터** | [4주차 열기](./week04/) |
| **Week 05** | **xPU를 어떻게 비교할 것인가** | GPU(SIMT) vs TPU(Systolic Array) vs NPU/LPU 아키텍처 및 워크로드별 비교 기준 | [5주차 열기](./week05/) |
| **Week 06** | **Python에서 Triton 커널로** | 1D 블록 포인터(`tl.arange`), 마스킹, Fused Online Softmax & RTX 4070 실측 | [6주차 열기](./week06/) |
| **Week 07** | **Tiled MatMul과 컴파일 과정** | 2D 타일 행렬곱, L2 그룹핑, **다단계 IR 탐색기(TTIR→SASS)**, A100 vs RTX 5090 비교 | [7주차 열기](./week07/) |
"""
    (DOCS_DIR / "index.md").write_text(index_md, encoding="utf-8")

    # 2. Build Overview Page (docs/intro/overview.md) from README.md
    readme_raw = (REPO_DIR / "README.md").read_text(encoding="utf-8")
    # Clean up links in README so `./week02` -> `../week02/`
    readme_cleaned = readme_raw.replace("(./week02)", "(/week02/)")
    readme_cleaned = readme_cleaned.replace("(./week03)", "(/week03/)")
    readme_cleaned = readme_cleaned.replace("(./week04)", "(/week04/)")
    readme_cleaned = readme_cleaned.replace("(./LICENSE)", "(https://github.com/legato-user-group/tensor2silicon-lab/blob/main/LICENSE)")
    readme_cleaned = sanitize_markdown_for_vitepress(readme_cleaned, week_num=1)
    (DOCS_DIR / "intro" / "overview.md").write_text(readme_cleaned, encoding="utf-8")

    # 3. Build Contributors Archive Matrix (docs/intro/contributors.md)
    contrib_lines = [
        "# 🧑‍🔬 연구원 아카이브 & 주차별 기여 매트릭스\n",
        "**Tensor2Silicon LAB**은 모든 연구원이 매주 직접 손계산, 코드 실습, 하드웨어 벤치마크, 교차 검증에 참여하여 함께 만들어가는 오픈 리서치 웹북입니다.\n",
        "## 👥 연구원 프로필\n",
        "| 프로필 | 이름 (핸들) | 주요 역할 및 관심 분야 | GitHub |",
        "| :---: | :--- | :--- | :---: |",
        "| <img src=\"https://github.com/yblee110.png\" width=\"56\" style=\"border-radius:50%\" /> | **이영빈 (`yblee`)** | 🧭 **랩장 (Lab Leader)** · 메인 교안 집필 및 커리큘럼 설계 | [@yblee110](https://github.com/yblee110) |",
        "| <img src=\"https://github.com/Jake-Song.png\" width=\"56\" style=\"border-radius:50%\" /> | **송민배 (`mbsong`)** | 🔬 A100 실측 IR 아티팩트 구축 · GPU/TPU 아키텍처 심층 비교 | [@Jake-Song](https://github.com/Jake-Song) |",
        "| <img src=\"https://github.com/SinGonKim.png\" width=\"56\" style=\"border-radius:50%\" /> | **김신곤 (`singon`)** | 🧮 정량적 손계산 해설 · 아키텍처 시각화 & RTX 4070 실측 | [@SinGonKim](https://github.com/SinGonKim) |",
        "| <img src=\"https://github.com/currybab.png\" width=\"56\" style=\"border-radius:50%\" /> | **박준 (`junpark`)** | 🔥 RTX 5090 (`sm_120`) PTX 실측 · 하드웨어 물리 계층 해부 | [@currybab](https://github.com/currybab) |",
        "| <img src=\"https://github.com/Kiminjo.png\" width=\"56\" style=\"border-radius:50%\" /> | **김인조 (`injokim`)** | 🖥️ 인터랙티브 HTML 시각화 제작 · 수식 유도 & MLIR 파이프라인 정리 | [@Kiminjo](https://github.com/Kiminjo) |",
        "| <img src=\"https://github.com/eloqlo.png\" width=\"56\" style=\"border-radius:50%\" /> | **이재형 (`jaehyeong`)** | 📐 핵심 개념 요약 · 컴파일 파이프라인 & 타일링 구조 정리 | [@eloqlo](https://github.com/eloqlo) |\n",
        "---\n",
        "## 📚 주차별 전체 문서 아카이브 매트릭스\n",
        "| 주차 | 📘 메인 교안 (`yblee`) | 🧭 안내 & 계산예제 | `mbsong` | `singon` | `junpark` | `injokim` | `jaehyeong` |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for w, meta in sorted(WEEKS_META.items()):
        r_map = {r["id"]: f"[열기](/week0{w}/researchers/{r['id']})" for r in meta["researchers"]}
        # Normalize kiminjo -> injokim for display column
        inj_link = r_map.get("injokim") or r_map.get("kiminjo") or "-"
        contrib_lines.append(
            f"| **Week 0{w}**<br/><small>{meta['title']}</small> "
            f"| [📘 메인 교안](/week0{w}/) "
            f"| [🧭 안내/예제](/week0{w}/guide) "
            f"| {r_map.get('mbsong', '-')} "
            f"| {r_map.get('singon', '-')} "
            f"| {r_map.get('junpark', '-')} "
            f"| {inj_link} "
            f"| {r_map.get('jaehyeong', '-')} |"
        )

    (DOCS_DIR / "intro" / "contributors.md").write_text("\n".join(contrib_lines) + "\n", encoding="utf-8")

    # 4. Build Weekly Chapters (Week 02 ~ Week 07)
    for w, meta in sorted(WEEKS_META.items()):
        week_dir = DOCS_DIR / f"week0{w}"
        res_dir = week_dir / "researchers"
        week_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)

        nav_bar = build_week_nav_bar_tag(w)
        synthesis_block = build_researcher_synthesis_section(w)
        interactive_block = build_week_interactive_section(w)

        # 4a. Main Lecture Note (docs/week0X/index.md)
        main_raw = meta["main_src"].read_text(encoding="utf-8")
        # Ensure top title starts with # if it started with ##
        if main_raw.lstrip().startswith("## "):
            main_raw = "#" + main_raw.lstrip()

        main_processed = wrap_solutions_in_details(main_raw)
        main_processed = sanitize_markdown_for_vitepress(main_processed, week_num=w, is_researcher=False)

        # Insert synthesis_block + interactive_block right before the first `## 1.` or after TOC `---`
        insert_match = re.search(r'\n##\s+1\.', main_processed)
        if insert_match:
            pos = insert_match.start()
            main_final = (
                nav_bar
                + main_processed[:pos]
                + "\n"
                + synthesis_block
                + interactive_block
                + main_processed[pos:]
            )
        else:
            main_final = nav_bar + synthesis_block + interactive_block + main_processed

        (week_dir / "index.md").write_text(main_final, encoding="utf-8")

        # 4b. Guide & Reference Calculation Examples (docs/week0X/guide.md)
        guide_parts = [nav_bar]
        if meta["guide_src"] and meta["guide_src"].exists():
            g_raw = meta["guide_src"].read_text(encoding="utf-8")
            if g_raw.lstrip().startswith("## "):
                g_raw = "#" + g_raw.lstrip()
            guide_parts.append(sanitize_markdown_for_vitepress(g_raw, week_num=w))

        if meta["ref_src"] and meta["ref_src"].exists():
            r_raw = meta["ref_src"].read_text(encoding="utf-8")
            r_raw = wrap_solutions_in_details(r_raw)
            guide_parts.append("\n\n---\n\n" + sanitize_markdown_for_vitepress(r_raw, week_num=w))

        (week_dir / "guide.md").write_text("\n".join(guide_parts), encoding="utf-8")

        # 4c. Researcher Notes (docs/week0X/researchers/<id>.md)
        for r in meta["researchers"]:
            r_parts = [
                nav_bar,
                f'> 🧑‍🔬 **연구원 리서치 노트** — 작성자: **{r["name"]}** (`{r["badge"]}`)\n>\n'
                f'> 💡 **핵심 기여 요약**: {r["summary"]}\n\n---\n\n'
            ]
            if "html_decks" in r and r["html_decks"]:
                r_parts.append(f"# {w}주차 인터랙티브 학습 노트 & 슬라이드 덱 ({r['name']})\n\n")
                for deck in r["html_decks"]:
                    r_parts.append(
                        f"## {deck['title']}\n\n"
                        f"{deck['desc']}\n\n"
                        f'<HtmlDeckViewer src="{deck["src"]}" title="{deck["title"]}" author="{r["name"]}" :height="640" />\n\n'
                    )
            for idx, fpath in enumerate(r["files"]):
                if fpath.exists():
                    f_raw = fpath.read_text(encoding="utf-8")
                    if f_raw.lstrip().startswith("## "):
                        f_raw = "#" + f_raw.lstrip()
                    if idx > 0:
                        r_parts.append("\n\n---\n\n")
                    r_parts.append(sanitize_markdown_for_vitepress(f_raw, week_num=w, is_researcher=True))

            (res_dir / f"{r['id']}.md").write_text("".join(r_parts), encoding="utf-8")

    print("Successfully generated all docs pages for T2S WebBook!")


if __name__ == "__main__":
    generate_all()
