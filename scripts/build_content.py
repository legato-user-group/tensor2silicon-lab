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
    # 0. Remove stray zero-width spaces (e.g., in week02 heading)
    md_text = md_text.replace("\u200b", "")

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
        "WeekNavBar", "HtmlDeckViewer", "RooflineCalculator", "IrPipelineViewer",
        "TldrMapCard", "MicroCheckpoint", "FeynmanPassport", "ConceptMatchQuest"
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


def insert_before_heading(md_text: str, heading_prefix: str, block: str, week_num: int) -> str:
    """
    Strictly inserts `block` right before `\\n{heading_prefix}` and raises an error if not found,
    guaranteeing no active-learning checkpoint is ever silently dropped.
    """
    target = "\n" + heading_prefix
    if target not in md_text:
        raise ValueError(f"[Week {week_num}] Heading prefix not found for checkpoint injection: {heading_prefix!r}")
    return md_text.replace(target, "\n" + block + heading_prefix, 1)


def build_tldr_map_card(week_num: int) -> str:
    if week_num == 2:
        return """
<TldrMapCard
  :week="2"
  intuition="Python 수식 한 줄(Y = ReLU(X @ W + b))은 곧바로 실리콘 위에서 돌지 않습니다. Eager 모드는 연산마다 커널을 따로 띄워 느린 HBM에 중간 결과를 계속 썼다 읽지만, 컴파일 스택(PyTorch 2 · OpenXLA · MAX)은 전체 연산 그래프(IR)를 먼저 캡처한 뒤 여러 연산을 단일 커널로 합쳐(Fusion) HBM 왕복을 제거합니다."
  :steps="[
    { tag: 'STEP 1 · 1~5절', title: '수식에서 그래프·IR 캡처와 Fusion', desc: '논리적 중간 텐서와 물리적 HBM 버퍼를 구분하고, Eager 실행과 그래프 컴파일러(Fusion)의 차이를 이해합니다.' },
    { tag: 'STEP 2 · 6~7절', title: '비동기 런타임과 GPU·TPU 실행', desc: 'CPU 호스트가 비동기 큐(Stream)에 커널을 등록하는 원리와 GPU(SIMT)·TPU(Systolic) 하드웨어 구조를 추적합니다.' },
    { tag: 'STEP 3 · 8~12절', title: '3대 실행 스택 & Triton 비교', desc: 'PyTorch(Dynamo·Inductor), JAX(OpenXLA), Modular MAX·Mojo 세 스택을 5대 계층 기준으로 대조합니다.' }
  ]"
  :keyQuestions="[
    '수식상 존재하는 중간 결과(Z = A + b)를 실제 HBM 메모리에 기록하지 않고도 다음 연산을 수행할 수 있는 이유는 무엇일까?',
    'GPU 커널 실행 시간을 잴 때 동기화(synchronize) 없이 CPU 타이머만 찍으면 왜 실제보다 훨씬 짧은 가짜 시간이 측정될까?',
    'PyTorch 2.0(Inductor), JAX(OpenXLA), Modular MAX는 각각 어느 계층에서 그래프를 캡처하고 커널 코드를 생성할까?'
  ]"
  :checkpoints="[
    { id: 'cp1-check', shortLabel: 'Eager vs Fusion 진단', anchorId: 'cp1-check' },
    { id: 'cp2-predict', shortLabel: '비동기 측정 함정', anchorId: 'cp2-predict' },
    { id: 'cp3-calc', shortLabel: '중간 버퍼 손계산', anchorId: 'cp3-calc' },
    { id: 'cp4-quest', shortLabel: '실행 스택 매칭 미션', anchorId: 'cp4-quest' },
    { id: 'cp5-feynman', shortLabel: '파인만 한줄설명 & 인증', anchorId: 'cp5-feynman' }
  ]"
/>

"""
    elif week_num == 3:
        return """
<TldrMapCard
  :week="3"
  intuition="커널의 실행 시간은 '얼마나 많이 계산하는가(F/P)'와 '얼마나 많이 옮기는가(Q/BW)' 중 더 오래 걸리는 쪽이 지배합니다. Vector Add처럼 원소당 계산이 1번뿐인 커널은 연산기가 아무리 빨라도 HBM 대역폭(BW)에 묶이므로, 여러 커널을 하나로 합쳐(Fusion) 불필요한 HBM 왕복 바이트(Q)를 줄여야 빨라집니다."
  :steps="[
    { tag: 'STEP 1 · 3~6절', title: '연산 시간 vs 메모리 이동 시간', desc: 'FLOPs/s(계산 속도)와 GB/s(대역폭), 메모리 용량과 대역폭을 구분하고 두 시간(T_compute, T_memory) 비교 모델을 세웁니다.' },
    { tag: 'STEP 2 · 7~8절', title: 'Vector Add & Kernel Fusion 손계산', desc: '단일 Vector Add의 병목을 숫자로 확인하고, 2단계 덧셈을 합칠 때 HBM 이동량이 24N → 16N Byte로 줄어드는 효과를 계산합니다.' },
    { tag: 'STEP 3 · 9~11절', title: '병목 판별과 개선 방향 선택', desc: '코드만 보고도 읽기·계산·쓰기 바이트를 세어 대역폭 병목인지 연산 병목인지 판별하는 습관을 익힙니다.' }
  ]"
  :keyQuestions="[
    'HBM 용량이 40 GB에서 80 GB로 2배 커진 GPU로 바꾸면, 대역폭(TB/s)이 같을 때 Vector Add 속도가 빨라질까?',
    '두 커널(T = X + Y, Z = T + W)을 하나로 합치면 총 덧셈 횟수(FLOPs)는 그대로인데 왜 실행 시간이 1/3이나 줄어들까?',
    '메모리 대역폭 병목에 걸린 연산을 연산 병목 구간으로 옮기려면 데이터 이동량(Q)을 어떻게 바꿔야 할까?'
  ]"
  :checkpoints="[
    { id: 'cp1-check', shortLabel: '용량 vs 대역폭 O/X', anchorId: 'cp1-check' },
    { id: 'cp2-predict', shortLabel: '커널 결합 결과 예측', anchorId: 'cp2-predict' },
    { id: 'cp3-calc', shortLabel: 'Fusion 배수 계산', anchorId: 'cp3-calc' },
    { id: 'cp4-quest', shortLabel: '병목 전환 시뮬레이터', anchorId: 'cp4-quest' },
    { id: 'cp5-feynman', shortLabel: '파인만 한줄설명 & 인증', anchorId: 'cp5-feynman' }
  ]"
/>

"""
    elif week_num == 4:
        return """
<TldrMapCard
  :week="4"
  intuition="행렬곱의 총 연산 횟수(2N³)는 단 1번도 줄일 수 없지만, 느린 HBM에서 가져온 입력 조각을 빠른 온칩 메모리(Shared Memory·레지스터)에 올려두고 여러 출력 계산에 재사용(Tiling)하면 HBM 왕복량이 1/t로 줄어들어 커널이 수십 배 빨라집니다."
  :steps="[
    { tag: 'STEP 1 · 3~5절', title: '산술 강도와 Roofline 읽기', desc: '1 Byte를 옮길 때 몇 FLOP을 계산하는지(I = F/Q) 구하고, 대역폭 병목과 연산 병목의 경계(분기점 I*)를 찾습니다.' },
    { tag: 'STEP 2 · 6~8절', title: 'Tiling 재사용 손계산', desc: '출력 타일(BM×BN)을 고정하고 K축으로 누적할 때 왜 HBM 읽기량이 8N³/t로 줄어드는지 손으로 직접 세어 봅니다.' },
    { tag: 'STEP 3 · 9~11절', title: '온칩 메모리 한계와 BK 분할', desc: '타일을 무조건 키울 수 없는 두 가지 물리적 한계(SRAM 용량, SM 병렬성)와 K축 폭(BK) 조절 원리를 이해합니다.' }
  ]"
  :keyQuestions="[
    '산술 강도(I = F/Q)가 하드웨어 분기점(I* = P/BW)보다 왼쪽에 있을 때, 연산 유닛(P)만 2배 빠른 칩으로 바꾸면 왜 속도가 그대로일까?',
    '4096×4096 행렬곱에서 타일 크기를 키울 때 연산량(F)은 그대로인데 어떻게 성능 상한이 올라갈까?',
    '출력 타일(BM×BN)은 크게 유지하면서 온칩 Shared Memory 사용량만 줄이려면 어떤 축의 폭을 조절해야 할까?'
  ]"
  :checkpoints="[
    { id: 'cp1-roofline', shortLabel: '오개념 진단 O/X', anchorId: 'cp1-roofline' },
    { id: 'cp2-predict', shortLabel: '타일링 결과 예측', anchorId: 'cp2-predict' },
    { id: 'cp3-calc', shortLabel: '손계산 숫자 빈칸', anchorId: 'cp3-calc' },
    { id: 'cp4-quest', shortLabel: '시뮬레이터 미션', anchorId: 'cp4-quest' },
    { id: 'cp5-feynman', shortLabel: '파인만 한줄설명 & 인증', anchorId: 'cp5-feynman' }
  ]"
/>

"""
    elif week_num == 5:
        return """
<TldrMapCard
  :week="5"
  intuition="가속기(GPU·TPU·NPU/LPU) 사양표에서 가장 큰 TFLOPs 숫자 하나만 비교하면 반드시 틀립니다. 자료형·희소성(Sparsity) 조건을 맞춘 실효 연산 성능(P)과 메모리 대역폭(BW), 그리고 워크로드의 산술 강도(Prefill vs Decode)를 함께 놓아야 어떤 칩이 실제로 빠른지 판별할 수 있습니다."
  :steps="[
    { tag: 'STEP 1 · 3~5절', title: '아키텍처와 사양표 조건 해독', desc: 'GPU(SIMT)·TPU(Systolic)·NPU의 구조 차이와 사양표 숫자 뒤에 숨은 정밀도·희소성 전제 조건을 해독합니다.' },
    { tag: 'STEP 2 · 6~8절', title: '장치 A vs B 역전 현상 손계산', desc: '피크 연산이 2배 낮은 장치가 대역폭 병목 작업에서 오히려 2배 빨라지는 조건과 호스트 전송 오버헤드를 계산합니다.' },
    { tag: 'STEP 3 · 9~11절', title: 'LLM 지표(TTFT·TPOT)와 실전 비교', desc: 'Prefill(연산 집약)과 Decode(대역폭 집약) 단계별로 어떤 하드웨어 스펙이 결정적인지 연결합니다.' }
  ]"
  :keyQuestions="[
    '사양표의 「INT8 2:4 Sparsity 기준 TOPS」 숫자를 일반적인 FP16 조밀(Dense) 행렬곱 시간 계산에 그대로 쓰면 왜 4배나 틀릴까?',
    '피크 연산 성능이 100 TFLOPs/s인 장치 B가 200 TFLOPs/s인 장치 A보다 2배 빠르게 끝나는 작업은 어떤 특징을 가질까?',
    'LLM 서비스에서 첫 토큰 지연(TTFT)과 토큰당 생성 시간(TPOT)은 각각 하드웨어의 어느 능력에 좌우될까?'
  ]"
  :checkpoints="[
    { id: 'cp1-check', shortLabel: '사양표 함정 O/X', anchorId: 'cp1-check' },
    { id: 'cp2-predict', shortLabel: '장치 승자 예측', anchorId: 'cp2-predict' },
    { id: 'cp3-calc', shortLabel: '실행 시간 손계산', anchorId: 'cp3-calc' },
    { id: 'cp4-quest', shortLabel: 'xPU·워크로드 매칭', anchorId: 'cp4-quest' },
    { id: 'cp5-feynman', shortLabel: '파인만 한줄설명 & 인증', anchorId: 'cp5-feynman' }
  ]"
/>

"""
    elif week_num == 6:
        return """
<TldrMapCard
  :week="6"
  intuition="CUDA가 개별 스레드(Thread) 단위로 인덱스를 짠다면, Triton은 1D/2D 블록(타일 벡터) 단위로 메모리를 읽고 온칩 SRAM 안에서 리덕션까지 한 번에 끝냅니다. Softmax 한 행 전체를 블록 하나에 담아 HBM 왕복을 8MN에서 2MN으로 1/4로 줄이는 것이 핵심입니다."
  :steps="[
    { tag: 'STEP 1 · 3~5절', title: '블록 인덱싱과 마스킹 기초', desc: 'program_id와 tl.arange로 포인터 블록을 만들고, 경계 밖 원소를 mask로 안전하게 처리하는 법을 익힙니다.' },
    { tag: 'STEP 2 · 6~8절', title: 'Fused Softmax와 패딩(-inf) 원리', desc: '분리 실행(8MN) 대비 결합 커널(2MN)의 데이터 이동 절감량과 최댓값 리덕션을 위한 -inf 패딩 이유를 파헤칩니다.' },
    { tag: 'STEP 3 · 9~11절', title: '정확도 검증과 실효 대역폭 측정', desc: 'torch.allclose 검증 기준과 워밍업·동기화를 갖춘 정확한 GPU 커널 벤치마크 방법을 확인합니다.' }
  ]"
  :keyQuestions="[
    'Triton으로 직접 짠 Vector Add 커널은 왜 PyTorch 기본 x + y 연산보다 드라마틱하게 빨라지지 않을까?',
    'Softmax 입력 행의 모든 원소가 음수일 때, 패딩 마스크 기본값(other)에 0.0을 넣으면 왜 계산 결과가 완전히 망가질까?',
    '크기 [M, N] 행렬의 Softmax를 단일 Triton 커널로 결합하면 주요 HBM 데이터 이동량이 몇 배 줄어들까?'
  ]"
  :checkpoints="[
    { id: 'cp1-check', shortLabel: 'Vector Add 한계 진단', anchorId: 'cp1-check' },
    { id: 'cp2-predict', shortLabel: '패딩 버그 찾기', anchorId: 'cp2-predict' },
    { id: 'cp3-calc', shortLabel: 'Softmax 이동량 계산', anchorId: 'cp3-calc' },
    { id: 'cp4-quest', shortLabel: 'Triton 코드 매칭', anchorId: 'cp4-quest' },
    { id: 'cp5-feynman', shortLabel: '파인만 한줄설명 & 인증', anchorId: 'cp5-feynman' }
  ]"
/>

"""
    elif week_num == 7:
        return """
<TldrMapCard
  :week="7"
  intuition="4주차에 손으로 유도한 2D 타일 재사용 수식은 Triton에서 단 10줄의 K축 포인터 전진 루프로 구현됩니다. 여기에 블록 실행 순서(Grouped Ordering)로 L2 캐시 적중률을 높이고, 컴파일러가 TTIR → TTGIR → PTX → SASS 단계를 거치며 비동기 복사(cp.async)와 Tensor Core(mma.sync) 기계어로 변환합니다."
  :steps="[
    { tag: 'STEP 1 · 3~4절', title: '2D 포인터 블록과 L2 Grouped Ordering', desc: '브로드캐스팅으로 2D 타일 주소를 만들고, 프로그램 실행 순서 재배치만으로 L2 캐시 재사용을 높이는 원리를 배웁니다.' },
    { tag: 'STEP 2 · 5~6절', title: 'Autotune과 다중 버퍼 SRAM 손계산', desc: '타일 후보별 산술 강도(Roofline)와 파이프라인 스테이지(num_stages)에 따른 Shared Memory 용량 한계를 직접 계산합니다.' },
    { tag: 'STEP 3 · 7~8절', title: '다단계 IR(TTIR→SASS) 실측 해부', desc: 'Python 커널이 어느 IR 단계에서 워프 레이아웃(#mma)과 하드웨어 기계어(HMMA)로 구체화되는지 추적합니다.' }
  ]"
  :keyQuestions="[
    '타일 크기는 똑같은데 프로그램 실행 순서만 Grouped Ordering으로 바꿨을 뿐인데 왜 HBM 트래픽이 줄어들까?',
    '출력 타일(BM×BN)을 고정한 채 K축 폭(BK)과 num_stages를 키우면 산술 강도와 Shared Memory 사용량은 각각 어떻게 변할까?',
    'Triton 컴파일 파이프라인 6단계 중 텐서에 #mma / #shared 하드웨어 레이아웃과 비동기 복사가 처음 도입되는 단계는 어디일까?'
  ]"
  :checkpoints="[
    { id: 'cp1-check', shortLabel: 'L2 그룹핑 O/X', anchorId: 'cp1-check' },
    { id: 'cp2-predict', shortLabel: 'BK 변화 결과 예측', anchorId: 'cp2-predict' },
    { id: 'cp3-calc', shortLabel: '파이프라인 SRAM 계산', anchorId: 'cp3-calc' },
    { id: 'cp4-quest', shortLabel: 'IR 파이프라인 탐정 미션', anchorId: 'cp4-quest' },
    { id: 'cp5-feynman', shortLabel: '파인만 한줄설명 & 인증', anchorId: 'cp5-feynman' }
  ]"
/>

"""
    return ""


def inject_week02_active_learning(md_text: str) -> str:
    cp1 = """
<MicroCheckpoint
  :week="2"
  checkpointId="cp1-check"
  mode="choice"
  badge="CHECKPOINT 1 · 2~5절 핵심 개념 점검 (Eager vs Fusion)"
  title="수식 속 중간 텐서는 항상 HBM 메모리에 저장되어야 할까?"
  question="2.3절의 공통 예제 A = X @ W → Z = A + b → Y = max(Z, 0)을 실행할 때, 기본 Eager 실행과 컴파일러 커널 퓨전(Fusion)의 차이에 대한 설명으로 가장 정확한 것은 무엇일까요?"
  :options="[
    {
      label: '커널 퓨전을 적용하면 행렬곱(X @ W) 자체의 곱셈·덧셈 횟수(FLOPs)가 절반으로 줄어들어 빨라진다.',
      correct: false,
      feedback: '아닙니다! 커널을 합쳐도 수학적 계산 횟수(FLOPs)는 변하지 않습니다. 줄어드는 것은 연산 횟수가 아니라 HBM 메모리 왕복입니다.'
    },
    {
      label: 'Eager 모드는 연산마다 별도 커널을 띄워 중간 텐서(A, Z)를 HBM에 썼다 다시 읽지만, Fusion은 온칩 레지스터/SRAM 위에서 연속 처리해 중간 HBM 왕복을 없앤다.',
      correct: true,
      feedback: '정확합니다! 수식상의 논리적 중간 결과(A, Z)를 물리적 HBM 버퍼에 실체화(Materialization)하지 않고 칩 내부 레지스터에서 바로 + b와 ReLU까지 끝내는 것이 Epilogue/Kernel Fusion의 핵심입니다.'
    },
    {
      label: 'Eager 모드에서도 Python 인터프리터가 자동으로 다음 줄 코드를 미리 읽어 단일 GPU 커널로 합쳐서 실행한다.',
      correct: false,
      feedback: 'Python Eager 실행은 한 줄씩 즉시 디스패치하므로 다음 줄에 어떤 연산이 올지 미리 알 수 없습니다. 그래서 전체 그래프(IR) 캡처 단계가 필요합니다!'
    }
  ]"
/>

"""
    cp2 = """
<MicroCheckpoint
  :week="2"
  checkpointId="cp2-predict"
  mode="choice"
  badge="CHECKPOINT 2 · 6절 비동기 실행 함정 진단 (Spot-the-Bug)"
  title="GPU 커널이 0.03 ms 만에 끝났다고요? 벤치마크 코드의 버그를 찾아보세요!"
  question="동료가 PyTorch에서 거대한 행렬곱 y = x @ w 의 실행 시간을 재기 위해 start = time.perf_counter(); y = x @ w; elapsed = time.perf_counter() - start 만 실행했더니 0.03 ms가 나왔다고 기뻐합니다. 실제로 무슨 일이 일어난 걸까요?"
  :options="[
    {
      label: 'GPU 연산기가 워낙 빨라서 실제로 4096×4096 행렬곱 계산이 0.03 ms 만에 완전히 끝난 것이다.',
      correct: false,
      feedback: '아직 실제 GPU 실리콘에서는 계산이 한창 진행 중이거나 막 시작된 상태일 가능성이 높습니다!'
    },
    {
      label: '비동기 실행(Asynchronous Launch) 때문에 CPU가 런타임 큐(Stream)에 커널 실행 요청만 등록하고 즉시 다음 줄로 넘어간 시간만 측정된 것이다.',
      correct: true,
      feedback: '정답입니다! GPU 커널 디스패치는 기본적으로 비동기이므로, 실제 실리콘 실행 완료 시간을 재려면 반드시 앞뒤에 torch.cuda.synchronize() (또는 CUDA Event)를 넣어 큐가 비워질 때까지 기다려야 합니다.'
    }
  ]"
/>

"""
    cp3 = """
<MicroCheckpoint
  :week="2"
  checkpointId="cp3-calc"
  mode="number"
  badge="CHECKPOINT 3 · Epilogue Fusion 중간 버퍼 이동량 계산"
  title="중간 텐서 실체화(Materialization)를 없애면 몇 MB를 아낄 수 있을까?"
  question="공통 예제 A = X @ W → Z = A + b → Y = max(Z, 0)에서 출력과 동일한 모양인 텐서 A, Z, Y의 크기가 각각 16 MB라고 가정합시다. Eager 모드에서는 A 쓰기(16) + A 읽기(16) + Z 쓰기(16) + Z 읽기(16)로 중간 결과 때문에만 총 64 MB의 HBM 왕복이 발생합니다. 만약 컴파일러가 Epilogue Fusion으로 행렬곱 누산 레지스터 위에서 + b와 ReLU까지 한 번에 처리하고 최종 Y(16 MB)만 HBM에 기록한다면, 중간 텐서(A, Z) 때문에 발생하는 HBM 왕복 데이터양은 몇 MB가 될까요?"
  :expectedAnswer="0"
  :tolerance="0.01"
  unit="MB"
  placeholder="숫자 입력"
  hint="중간 결과 A와 Z가 HBM에 아예 기록되지도, 다시 읽히지도 않고 칩 내부 레지스터에서만 소비된 뒤 사라집니다!"
  solutionExplanation="정답입니다! 중간 텐서 A와 Z의 HBM 왕복량은 정확히 0 MB가 됩니다(64 MB 절감!). 논리적으로는 수식에 A와 Z가 존재하지만 물리적 HBM 메모리에는 최종 결과 Y만 기록됩니다."
/>

"""
    cp4 = """
## 🖥️ 실습 검증: 실행 스택 매칭 미션 & 인터랙티브 슬라이드 (Checkpoint 4)

<ConceptMatchQuest
  :week="2"
  checkpointId="cp4-quest"
  badge="CHECKPOINT 4 · 5대 실행 스택 역할 매칭 미션"
  title="방금 비교한 실행 스택 구성 요소와 핵심 역할을 바르게 연결해 보세요!"
  instruction="왼쪽의 기술/계층 카드를 먼저 클릭한 뒤, 오른쪽에서 알맞은 설명 카드를 클릭해 4쌍을 모두 연결하면 미션이 완료됩니다."
  :pairs="[
    {
      id: 'dynamo',
      left: 'TorchDynamo (PyTorch 2)',
      right: 'Python 바이트코드를 가로채 실행 연산들을 FX 그래프(IR)로 캡처하는 프론트엔드'
    },
    {
      id: 'inductor',
      left: 'TorchInductor',
      right: '캡처된 그래프에서 연산 퓨전을 수행하고 Triton 또는 C++ 커널 코드를 자동 생성하는 백엔드 컴파일러'
    },
    {
      id: 'stablehlo',
      left: 'StableHLO (JAX · OpenXLA)',
      right: '다양한 프레임워크와 하드웨어 컴파일러 사이를 연결하는 이식성 높은 고수준 MLIR 연산 집합'
    },
    {
      id: 'stream',
      left: 'CUDA Stream / Runtime',
      right: '호스트(CPU)가 디바이스(GPU)에 커널을 순서대로 비동기 등록하고 실행을 중재하는 실행 큐 계층'
    }
  ]"
/>

### 🎞️ 인터랙티브 실행 스택 아키텍처 & 슬라이드 덱 (`injokim`)

> 2주차 연구원(`injokim`)이 제작한 인터랙티브 학습 노트와 `torch.compile` 내부 구조 슬라이드 덱입니다. 직접 탭을 클릭하거나 **전체화면**으로 열람해 보세요.

<HtmlDeckViewer src="/decks/injokim-week02.html" title="PyTorch · OpenXLA · Modular MAX 실행 스택 인터랙티브 노트" author="injokim (김인조)" :height="560" />

<HtmlDeckViewer src="/decks/torch-compile-anatomy-deck.html" title="torch.compile 내부 구조 해부 (Dynamo → AOTAutograd → Inductor)" author="T2S Lab" :height="560" />

---

"""
    cp5 = """
## 🏅 Week 02 완독 점검 & 인증 패스포트 (Checkpoint 5)

<FeynmanPassport
  :week="2"
  weekTitle="모델에서 실리콘까지 — AI 실행 스택 5대 계층 해부"
  checkpointId="cp5-feynman"
  :allCheckpointIds="['cp1-check', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman']"
  promptQuestion="질문: PyTorch Eager 실행과 비교했을 때, torch.compile이나 OpenXLA처럼 그래프(IR)를 먼저 캡처해 컴파일하는 방식이 실제 가속기에서 더 빠르게 동작하는 가장 핵심적인 이유는 무엇인가요?"
  modelAnswer="Eager 실행은 연산을 하나씩 즉시 디스패치하므로 중간 텐서를 매번 느린 HBM에 썼다가 다시 읽고 커널 런치 오버헤드도 반복됩니다. 반면 그래프(IR)를 먼저 캡처하면 컴파일러가 앞뒤 연산 흐름을 한눈에 보고 여러 연산을 단일 커널로 결합(Fusion)하여, 중간 결과를 온칩 레지스터·SRAM 안에서 바로 재사용하고 HBM 왕복을 제거할 수 있습니다."
  :selfCheckItems="[
    'Eager 모드의 한계 언급: 연산별 개별 커널 실행으로 인한 중간 결과의 HBM 읽기/쓰기 반복(또는 런치 오버헤드)을 짚었다.',
    '그래프 컴파일의 이점 언급: 앞뒤 연산을 함께 보고 커널 결합(Fusion)을 수행해 온칩 메모리에서 바로 이어 계산한다는 점을 설명했다.'
  ]"
/>

---

"""
    md_text = insert_before_heading(md_text, "## 6. 실행 요청에서 커널 실행으로: 런타임과 비동기성", cp1, 2)
    md_text = insert_before_heading(md_text, "## 7. 커널에서 칩 내부로: GPU와 TPU", cp2, 2)
    md_text = insert_before_heading(md_text, "## 8. PyTorch 경로와 예제", cp3, 2)
    md_text = insert_before_heading(md_text, "## 13. 작게 해볼 수 있는 관찰 실험", cp4, 2)
    md_text = insert_before_heading(md_text, "## 15. 용어 사전과 최종 요약", cp5, 2)
    return md_text


def inject_week03_active_learning(md_text: str) -> str:
    cp1 = """
<MicroCheckpoint
  :week="3"
  checkpointId="cp1-check"
  mode="choice"
  badge="CHECKPOINT 1 · 4~6절 오개념 함정 진단 (O/X)"
  title="HBM 메모리 용량(GB)이 2배 큰 GPU로 바꾸면 Vector Add도 2배 빨라질까?"
  question="동료가 '기존 40 GB HBM GPU 대신 80 GB HBM을 탑재한 모델로 업그레이드했으니(단, 메모리 대역폭 BW = 1.0 TB/s와 연산 성능 P는 동일), 이제 Vector Add 커널 실행 시간도 절반으로 줄어들 거야!'라고 말합니다. 이 주장은 맞을까요?"
  :options="[
    {
      label: '⭕ 맞다 — HBM 용량이 2배 넓어졌으므로 데이터를 주고받는 속도도 2배 빨라진다.',
      correct: false,
      feedback: '4절의 핵심 구분을 떠올려 보세요! 용량(GB)은 창고의 크기이고, 대역폭(TB/s)은 초당 지나갈 수 있는 도로의 폭입니다.'
    },
    {
      label: '❌ 틀리다 — 용량(80 GB)은 더 큰 모델/배치를 올릴 수 있게 해줄 뿐, 대역폭(BW = 1.0 TB/s)이 같다면 데이터 이동 시간(Q / BW)은 그대로다.',
      correct: true,
      feedback: '정확합니다! 메모리가 부족해 실행 자체가 안 되던 문제를 해결하는 것(용량)과, 초당 몇 바이트를 실어 나르느냐(대역폭)는 완전히 다른 축입니다. 바로 아래 7절에서 실제 숫자로 계산해 봅시다!'
    }
  ]"
/>

"""
    cp2 = """
<MicroCheckpoint
  :week="3"
  checkpointId="cp2-predict"
  mode="choice"
  badge="CHECKPOINT 2 · 8절 읽기 전 결과 예측 (Predict-Before-Read)"
  title="8절을 읽기 전에 예측해 보세요: 두 벡터 덧셈 커널을 하나로 합치면 무엇이 줄어들까요?"
  question="길이 N인 FP32(4 Byte) 벡터들에 대해 ① T = X + Y 실행 후 ② Z = T + W 를 따로 실행하던 코드를, 단일 커널 Z = (X + Y) + W 로 결합(Fusion)하려 합니다. 이때 총 덧셈 연산량(F)과 HBM 메모리 이동량(Q)은 어떻게 변할까요?"
  :options="[
    {
      label: '덧셈 연산량(F)이 2N에서 1N으로 절반 줄어들고, HBM 이동량(Q)은 24N Byte로 유지된다.',
      correct: false,
      feedback: '세 벡터(X, Y, W)를 더하려면 원소마다 덧셈은 여전히 두 번씩(총 2N FLOPs) 수행해야 합니다!'
    },
    {
      label: '덧셈 연산량(F = 2N)은 그대로지만, 중간 벡터 T를 HBM에 쓰고(4N) 다시 읽는(4N) 과정이 사라져 HBM 이동량이 24N → 16N Byte로 줄어든다.',
      correct: true,
      feedback: '완벽한 예측입니다! 연산 횟수(2N)는 똑같지만 중간 결과 T의 왕복(8N Byte)이 제거되어 전체 메모리 이동량이 1/3이나 감소합니다. 바로 아래 8절 손계산으로 확인해 보세요!'
    },
    {
      label: '연산량(F)과 HBM 이동량(Q) 모두 변하지 않고 오직 CPU 커널 런치 오버헤드만 줄어든다.',
      correct: false,
      feedback: '런치 오버헤드도 줄어들지만, 원소 수가 클 때 훨씬 결정적인 이득은 중간 텐서 T의 HBM 쓰기+읽기(8N Byte)가 통째로 사라지는 것입니다!'
    }
  ]"
/>

"""
    cp3 = """
<MicroCheckpoint
  :week="3"
  checkpointId="cp3-calc"
  mode="number"
  badge="CHECKPOINT 3 · 8절 Kernel Fusion 속도 향상 배수 검증"
  title="24N Byte에서 16N Byte로 줄어들면 이상적 실행 속도는 몇 배 빨라질까?"
  question="위 8절 계산에서 N = 10⁸, FP32(4 Byte), HBM 대역폭 BW = 1 TB/s일 때, 분리 실행(Q = 24N Byte = 2.4 GB)의 이상적 메모리 시간은 2.4 ms였고, 커널 결합 실행(Q = 16N Byte = 1.6 GB)의 이상적 메모리 시간은 1.6 ms였습니다. 커널 결합을 통해 이상적 실행 속도는 몇 배(2.4 ms ÷ 1.6 ms) 빨라졌나요?"
  :expectedAnswer="1.5"
  :tolerance="0.02"
  unit="배"
  placeholder="소수 첫째 자리 (예: 1.5)"
  hint="분리 실행 시간(2.4 ms)을 결합 실행 시간(1.6 ms)으로 나누어 보세요: 2.4 / 1.6 = 24 / 16 !"
  solutionExplanation="정확합니다! 24N / 16N = 1.5배 빨라집니다(실행 시간 기준으로는 2.4 ms → 1.6 ms로 33.3% 단축). 대역폭 병목 커널에서는 HBM 이동 바이트 비율이 곧바로 실행 시간 비율을 결정합니다!"
/>

"""
    cp4 = """
## 🎛️ 실습 검증: 메모리 병목에서 연산 병목으로 탈출하기 (Checkpoint 4)

> 원소별 연산(Vector Add)은 아무리 묶어도 한 번 읽은 데이터로 계산을 몇 번밖에 하지 못하지만, 다음 주차로 이어지는 **행렬곱(MatMul)** 은 온칩 메모리에 올린 타일 크기를 키울수록 데이터 재사용이 늘어 **어느 순간 메모리 대역폭 병목에서 연산 병목으로 전환**됩니다. 아래 시뮬레이터에서 직접 타일 크기를 눌러 그 전환 지점을 찾아보세요!

<RooflineCalculator :week="3" checkpointId="cp4-quest" questType="week03" />

---

"""
    cp5 = """
## 🏅 Week 03 완독 점검 & 인증 패스포트 (Checkpoint 5)

<FeynmanPassport
  :week="3"
  weekTitle="커널의 시간은 어디에서 소비되는가 — 연산 시간 vs 메모리 이동 시간"
  checkpointId="cp5-feynman"
  :allCheckpointIds="['cp1-check', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman']"
  promptQuestion="질문: Vector Add(Z = X + Y)처럼 단순한 원소별 커널은 GPU의 연산 성능(TFLOPs/s)을 높여도 거의 빨라지지 않습니다. 그 이유와 실제 실행 시간을 줄이는 핵심 방법을 '두 시간(T_compute, T_memory)' 관점에서 설명해 보세요."
  modelAnswer="Vector Add는 원소 하나당 HBM에서 12 Byte(읽기 8B + 쓰기 4B)를 옮기면서 덧셈은 단 1 FLOP만 수행하므로, 데이터 이동 시간(T_memory = Q/BW)이 연산 시간(T_compute = F/P)보다 수십 배 길어 대역폭 병목에 걸립니다. 따라서 연산기(P)를 키워도 T_memory에 막혀 빨라지지 않으며, 앞뒤 연산을 커널 결합(Fusion)해 중간 텐서의 HBM 왕복 데이터양(Q) 자체를 줄여야 실행 시간이 단축됩니다."
  :selfCheckItems="[
    '병목 원인 명시: 데이터 이동 시간(Q/BW)이 연산 시간(F/P)보다 훨씬 커서 메모리 대역폭 병목(Memory-Bound) 상태임을 설명했다.',
    '해결 방향 명시: 연산 성능(P)이 아니라 커널 결합(Fusion)이나 재사용으로 HBM 이동 바이트(Q)를 줄여야 한다는 점을 짚었다.'
  ]"
/>

---

"""
    md_text = insert_before_heading(md_text, "## 7. 계산 예제 1 — Vector Add의 시간", cp1, 3)
    md_text = insert_before_heading(md_text, "## 8. 계산 예제 2 — 커널을 합치면 무엇이 달라질까", cp2, 3)
    md_text = insert_before_heading(md_text, "## 9. 병목을 해석하고 개선 방향 고르기", cp3, 3)
    md_text = insert_before_heading(md_text, "## 10. 코드에서 읽기와 계산과 쓰기 찾기", cp4, 3)
    md_text = insert_before_heading(md_text, "## 16. 스터디 발표와 용어 정리", cp5, 3)
    return md_text


def inject_week04_active_learning(md_text: str) -> str:
    cp1_and_cp2 = """
<MicroCheckpoint
  :week="4"
  checkpointId="cp1-roofline"
  mode="choice"
  badge="CHECKPOINT 1 · 3~5절 오개념 함정 진단 (O/X)"
  title="연산 성능이 2배인 신형 GPU로 바꾸면 내 커널도 2배 빨라질까?"
  question="현재 하드웨어 조건(BW = 1 TB/s, P = 20 TFLOPs/s, 분기점 I* = 20)에서 산술 강도가 I = 8 FLOP/Byte인 커널을 실행 중입니다. 동료가 'HBM 대역폭(1 TB/s)은 동일하지만 피크 연산 성능만 P = 40 TFLOPs/s로 2배 높은 신형 칩으로 바꾸면 이 커널의 성능 상한도 2배 올라갈 거야!'라고 주장합니다. 이 주장은 맞을까요?"
  :options="[
    {
      label: '⭕ 맞다 — 하드웨어 피크 연산 성능(P)이 2배가 되었으므로 커널 상한도 8에서 16 TFLOPs/s로 올라간다.',
      correct: false,
      feedback: '5.4절 표를 다시 떠올려 보세요! P가 20→40으로 커지면 수평 지붕만 높아지고 분기점(I* = 40)이 오른쪽으로 멀어질 뿐, 대역폭 경사면(BW × I)은 그대로입니다.'
    },
    {
      label: '❌ 틀리다 — 현재 대역폭 병목 구간(I = 8 < 20)에 있으므로, BW나 I를 키우지 않고 P만 높여도 상한은 8 TFLOPs/s 그대로다.',
      correct: true,
      feedback: '정확합니다! R = min(P, BW × I)에서 min(40, 1 × 8) = 8 TFLOPs/s이므로, 데이터 이동(BW)에 막힌 커널은 연산 유닛만 늘려도 빨라지지 않습니다. 그래서 다음 6절의 Tiling(데이터 재사용으로 I 자체를 오른쪽으로 이동시키는 기법)이 필수적입니다!'
    }
  ]"
/>

<MicroCheckpoint
  :week="4"
  checkpointId="cp2-predict"
  mode="choice"
  badge="CHECKPOINT 2 · 6절 읽기 전 결과 예측 (Predict-Before-Read)"
  title="6절을 읽기 전에 먼저 예측해 보세요: 타일을 키우면 계산 횟수(FLOPs)가 줄어들까요?"
  question="4096×4096 행렬곱(C = A @ B)에서 출력 타일 크기를 32×32에서 128×128로 4배(면적 16배) 키우면, GPU가 수행하는 총 곱셈·덧셈 연산량(F, FLOPs)은 어떻게 변할까요?"
  :options="[
    {
      label: '타일 한 변이 4배 커졌으므로 총 연산량(F)도 1/4로 줄어든다.',
      correct: false,
      feedback: '아닙니다! 1/4로 줄어드는 것은 연산량(F)이 아니라 HBM 메모리 읽기량(Q_read)입니다.'
    },
    {
      label: '타일 면적이 16배 커졌으므로 총 연산량(F)이 1/16로 줄어든다.',
      correct: false,
      feedback: '타일은 계산 순서를 묶는 단위일 뿐, 결과 행렬 C의 각 원소를 구하는 내적 연산을 생략하지는 않습니다.'
    },
    {
      label: '총 연산량(F = 2N³)은 단 1 FLOP도 변하지 않고 완전히 똑같다.',
      correct: true,
      feedback: '정답입니다! Tiling은 계산 횟수(F)를 줄이는 마법이 아니라, 같은 F를 수행하면서 느린 HBM에서 데이터를 읽어오는 횟수(Q)만 줄여 산술 강도(I = F/Q)를 끌어올리는 기법입니다. 바로 아래 6~8절에서 그 구조를 확인해 보세요!'
    }
  ]"
/>

"""
    cp3_calc = """
<MicroCheckpoint
  :week="4"
  checkpointId="cp3-calc"
  mode="number"
  badge="CHECKPOINT 3 · 8절 손계산 1단계 변형 검증"
  title="타일 한 변(t)을 64에서 256으로 4배 키우면 입력 읽기량은 몇 GiB가 될까?"
  question="위 8.4~8.5절에서 N = 4096, FP32(4 Byte)일 때 전체 입력 읽기량 식은 Q_read = 8N³ / t 였고, t = 64일 때 Q_read = 8,589,934,592 Byte = 정확히 8 GiB(2³⁰ Byte 기준)였습니다. 만약 같은 조건에서 정사각 타일 크기만 t = 256으로 키운다면, 전체 입력 읽기량 Q_read는 몇 GiB가 될까요?"
  :expectedAnswer="2"
  :tolerance="0.01"
  unit="GiB"
  placeholder="정수 입력"
  hint="Q_read = 8N³ / t 에서 N은 고정되어 있고 분모 t만 64에서 256으로 4배 커졌습니다. 8 GiB ÷ 4를 계산해 보세요!"
  solutionExplanation="정확합니다! 분모 t가 64 → 256으로 4배 커지므로 HBM 입력 읽기량은 8 GiB → 2 GiB로 정확히 1/4로 줄어듭니다. (단, 8.6절에서 봤듯 t=128에서 이미 연산 상한 6.872 ms에 도달했으므로 t=256으로 더 키워도 이상적 실행 시간은 6.872 ms 아래로 내려가지 않습니다!)"
/>

"""
    cp4_simulator_quest = """
## 🎛️ 실습 검증: 인터랙티브 Roofline & Tiling 미션 (Checkpoint 4)

> 방금 10~11절에서 배운 **핵심 원리** — *"출력 타일(`BM×BN`)은 산술 강도를 결정하고, K 방향 폭(`BK`)과 파이프라인 단계는 온칩 Shared Memory 사용량을 좌우한다"* — 를 아래 시뮬레이터에서 직접 손으로 조작해 증명해 보세요!

<RooflineCalculator :week="4" checkpointId="cp4-quest" questType="week04" />

### 📊 4주차 핵심 시각화 도표 모음

| Roofline 성능 상한과 재사용 조건 | 4×4 행렬곱의 2×2 출력 타일 누적 과정 |
| :---: | :---: |
| ![Roofline 그래프](/assets/week04/roofline.svg) | ![Tiling 누적 과정](/assets/week04/tiling.svg) |

<details class="t2s-solution-toggle">
<summary>🖼️ 추가 시각 자료 펼치기 (타일 저장 공간·BK 폭 비교·하드웨어 변화에 따른 Roofline 이동)</summary>

![타일 저장 공간과 병렬성](/assets/week04/lecture-visuals/tile_storage_parallelism.svg)

![BK 폭 비교](/assets/week04/lecture-visuals/bk_width_comparison.svg)

![하드웨어 변화와 Roofline](/assets/week04/lecture-visuals/roofline_hardware_changes.svg)

</details>

---

"""
    cp5_passport = """
## 🏅 Week 04 완독 점검 & 인증 패스포트 (Checkpoint 5)

<FeynmanPassport
  :week="4"
  weekTitle="Roofline과 Tiling — 산술 강도와 온칩 데이터 재사용"
  checkpointId="cp5-feynman"
  :allCheckpointIds="['cp1-roofline', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman']"
  promptQuestion="질문: 타일 크기를 키울수록 HBM 이동량(Q)이 줄어 산술 강도가 높아지는데도, 실제 GPU 커널에서 타일 크기를 무한정 크게 잡을 수 없는 하드웨어적 이유 2가지는 무엇인가요?"
  modelAnswer="첫째, 타일이 커질수록 입력 타일 버퍼(Shared Memory)와 누산값(레지스터)이 요구하는 온칩 메모리 용량이 급증하여 SM당 하드웨어 한도를 초과하거나 레지스터 스필(Spill)이 발생합니다. 둘째, 전체 행렬을 큰 타일로 나누면 생성되는 총 출력 타일(블록/CTA) 개수가 줄어들고 블록당 자원 점유가 커져 여러 SM에 고르게 일을 나누는 병렬성(Occupancy)이 떨어집니다."
  :selfCheckItems="[
    '온칩 저장 공간 한계 언급: Shared Memory 또는 레지스터 파일 용량 한계(초과/스필 위험)를 설명했다.',
    '병렬성(일감 분배) 감소 언급: 총 출력 타일(CTA) 수가 줄거나 SM당 동시 실행 블록 수(Occupancy)가 줄어드는 점을 설명했다.'
  ]"
/>

---

"""
    md_text = insert_before_heading(md_text, "## 6. Tiling은 무엇을 나누고 무엇을 유지할까", cp1_and_cp2, 4)
    md_text = insert_before_heading(md_text, "## 9. 재사용 가정이 달라지면 산술 강도도 달라진다", cp3_calc, 4)
    md_text = insert_before_heading(md_text, "## 12. 선택 활동", cp4_simulator_quest, 4)
    md_text = insert_before_heading(md_text, "## 16. 스터디에서 설명하고 함께 점검하기", cp5_passport, 4)
    return md_text


def inject_week05_active_learning(md_text: str) -> str:
    cp1 = """
<MicroCheckpoint
  :week="5"
  checkpointId="cp1-check"
  mode="choice"
  badge="CHECKPOINT 1 · 3~4절 사양표 함정 진단 (O/X)"
  title="브로셔의 제일 큰 숫자(Sparsity 포함 TOPS)로 내 FP16 모델 시간을 계산해도 될까?"
  question="어떤 신형 가속기 홍보 자료에 「최대 성능 4,000 TOPS (INT8, 2:4 구조적 희소성 적용 기준)」이라고 적혀 있습니다. 내가 실행할 모델이 일반적인 조밀(Dense) FP16 행렬곱일 때, T_compute = F / P 식의 분모 P에 4,000을 그대로 대입해도 될까요?"
  :options="[
    {
      label: '⭕ 된다 — 같은 칩의 최대 연산 능력을 나타내는 대표 수치이므로 그대로 대입해 상한을 구한다.',
      correct: false,
      feedback: '4절의 핵심 경고를 확인해 보세요! 자료형(INT8 vs FP16)에서 보통 2배, 2:4 희소성(Sparsity) 조건에서 또 2배가 부풀려져 최대 4배나 과대평가하게 됩니다.'
    },
    {
      label: '❌ 안 된다 — 반드시 내가 실행할 작업과 동일한 자료형(FP16) 및 조밀(Dense) 조건의 실효 피크 연산 성능(P)을 찾아 대입해야 한다.',
      correct: true,
      feedback: '정확합니다! 사양표 숫자는 항상 「어떤 자료형인가, 희소성(Sparsity) 가정인가, 부스트 클록 기준인가」라는 조건표와 한 몸으로 읽어야 공정한 비교가 가능합니다.'
    }
  ]"
/>

"""
    cp2 = """
<MicroCheckpoint
  :week="5"
  checkpointId="cp2-predict"
  mode="choice"
  badge="CHECKPOINT 2 · 7절 읽기 전 승자 예측 (Predict-Before-Read)"
  title="7절을 읽기 전에 예측해 보세요: 피크 연산 성능이 절반인 장치 B가 장치 A를 이길 수 있을까요?"
  question="장치 A(P = 200 TFLOPs/s, BW = 1 TB/s)와 장치 B(P = 100 TFLOPs/s, BW = 2 TB/s)가 있습니다. 산술 강도가 낮은 작업 1(F = 10¹² FLOPs, Q = 100 GB, 즉 I = 10 FLOP/Byte)을 두 장치에서 실행하면 어느 쪽이 얼마나 더 빠를까요?"
  :options="[
    {
      label: '피크 연산 성능이 2배 높은 장치 A가 2배 더 빠르다.',
      correct: false,
      feedback: '장치 A의 분기점은 I* = 200 / 1 = 200이고, 작업 1의 산술 강도는 I = 10입니다! 연산기가 아니라 메모리 대역폭이 발목을 잡고 있습니다.'
    },
    {
      label: '연산 성능은 절반이지만 메모리 대역폭이 2배 넓은 장치 B가 오히려 2배 더 빠르다 (50 ms vs 100 ms).',
      correct: true,
      feedback: '정답입니다! I = 10인 작업은 두 장치 모두에서 대역폭 병목이므로 T_memory = Q / BW가 시간을 결정합니다. 따라서 BW가 2 TB/s인 장치 B(50 ms)가 장치 A(100 ms)를 2배 차이로 이깁니다!'
    },
    {
      label: 'P × BW 곱이 서로 같으므로 두 장치의 실행 시간은 완전히 동일하다.',
      correct: false,
      feedback: '실행 시간은 P × BW가 아니라 max(F/P, Q/BW)로 결정됩니다! 아래 7절 계산표에서 직접 확인해 보세요.'
    }
  ]"
/>

"""
    cp3 = """
<MicroCheckpoint
  :week="5"
  checkpointId="cp3-calc"
  mode="number"
  badge="CHECKPOINT 3 · 7절 작업 2(연산 집약) 실행 시간 검증"
  title="작업의 산술 강도가 I = 1000으로 바뀌면 장치 B의 실행 시간은 몇 ms가 될까?"
  question="위 7절의 장치 B(P = 100 TFLOPs/s = 10¹⁴ FLOPs/s, BW = 2 TB/s)에서 이번에는 데이터 이동량은 똑같이 Q = 100 GB이지만 연산량이 100배 많은 작업 2(F = 10¹⁴ FLOPs, I = 1000 FLOP/Byte)를 실행합니다. 메모리 시간은 T_memory = 50 ms이지만 이제는 연산 시간(T_compute = F / P)이 병목이 됩니다. 장치 B의 이상적 실행 시간 T_ideal = max(T_compute, T_memory)은 몇 ms일까요?"
  :expectedAnswer="1000"
  :tolerance="1"
  unit="ms"
  placeholder="정수 입력 (단위: ms)"
  hint="T_compute = 10¹⁴ FLOPs ÷ (100 × 10¹² FLOPs/s) = 1.0 초(s)입니다. 1.0 초를 밀리초(ms)로 변환해 보세요!"
  solutionExplanation="정확합니다! T_compute = 1.0 s = 1000 ms이고 T_memory = 50 ms이므로 max(1000, 50) = 1000 ms가 됩니다(반면 P=200인 장치 A는 500 ms 만에 끝납니다!). 즉, 어느 가속기가 더 빠른지는 고정된 서열이 아니라 작업의 산술 강도(I)에 따라 정반대로 뒤집힙니다."
/>

"""
    cp4 = """
## 🧩 실습 검증: xPU 아키텍처 & LLM 워크로드 매칭 미션 (Checkpoint 4)

<ConceptMatchQuest
  :week="5"
  checkpointId="cp4-quest"
  badge="CHECKPOINT 4 · xPU 구조 & LLM 지표 연결 미션"
  title="가속기 아키텍처와 LLM 추론 단계별 병목 특성을 바르게 연결해 보세요!"
  instruction="왼쪽의 하드웨어/워크로드 항목을 클릭한 뒤, 오른쪽에서 가장 알맞은 구조적·성능적 특징을 선택해 4쌍을 모두 연결하세요."
  :pairs="[
    {
      id: 'gpu',
      left: 'GPU (SIMT + Tensor Core)',
      right: '수많은 워프(스레드 묶음) 스케줄링과 캐시/Shared Memory로 불규칙 분기와 범용 병렬성을 유연하게 처리'
    },
    {
      id: 'tpu',
      left: 'TPU (Systolic Array + VMEM)',
      right: '거대한 2D 맥동 배열(MXU)과 컴파일러가 직접 관리하는 대용량 스크래치패드(VMEM)로 정형 행렬 연산 극대화'
    },
    {
      id: 'prefill',
      left: 'LLM Prefill 단계 (TTFT 결정)',
      right: '긴 입력 프롬프트를 한꺼번에 행렬곱으로 처리해 산술 강도(I)가 높고 연산 성능(TFLOPs/s)이 핵심인 구간'
    },
    {
      id: 'decode',
      left: 'LLM Decode 단계 (TPOT 결정)',
      right: '토큰을 1개씩 생성할 때마다 거대한 가중치와 KV 캐시를 읽어야 해 메모리 대역폭(TB/s)에 묶이는 구간'
    }
  ]"
/>

---

"""
    cp5 = """
## 🏅 Week 05 완독 점검 & 인증 패스포트 (Checkpoint 5)

<FeynmanPassport
  :week="5"
  weekTitle="xPU를 어떻게 비교할 것인가 — 아키텍처·사양표·워크로드 비교 기준"
  checkpointId="cp5-feynman"
  :allCheckpointIds="['cp1-check', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman']"
  promptQuestion="질문: 피크 연산 성능(TFLOPs/s)은 더 낮지만 메모리 대역폭(TB/s)이 넓은 가속기가 LLM 추론의 Decode(토큰 순차 생성) 단계에서 오히려 더 빠른 토큰 생성 속도(TPOT)를 내는 이유를 설명해 보세요."
  modelAnswer="LLM의 Decode 단계는 한 번에 토큰 1개(또는 작은 배치)를 생성하기 위해 모델 전체 가중치와 KV 캐시를 메모리에서 읽어와야 하므로 데이터 이동량 대비 연산량인 산술 강도(I)가 매우 낮습니다. 따라서 하드웨어 분기점 왼쪽의 메모리 대역폭 병목 구간에 놓이게 되며, 피크 연산 성능(P)보다 초당 데이터를 실어 나르는 메모리 대역폭(BW)이 큰 장치가 더 짧은 TPOT를 달성합니다."
  :selfCheckItems="[
    'Decode 단계의 낮은 산술 강도 언급: 토큰당 읽어야 할 가중치·KV 캐시 크기에 비해 연산량이 적어 Memory-Bound 구간임을 설명했다.',
    '대역폭(BW) 지배 원리 언급: 대역폭 병목에서는 피크 TFLOPs(P)보다 메모리 대역폭(Q/BW)이 토큰당 지연 시간(TPOT)을 좌우함을 짚었다.'
  ]"
/>

---

"""
    md_text = insert_before_heading(md_text, "## 5. 소프트웨어와 실행 경로 — 같은 모델이 가속기까지 내려가는 과정", cp1, 5)
    md_text = insert_before_heading(md_text, "## 7. 계산 예제 1 — 연산 성능이 더 큰 장치가 항상 빠를까", cp2, 5)
    md_text = insert_before_heading(md_text, "## 8. 계산 예제 2 — 장치 연산이 빠르면 전체 응답도 빠를까", cp3, 5)
    md_text = insert_before_heading(md_text, "## 10. 장비 없이 참여하기 — 무료 실습 환경, 해석기, 시뮬레이터", cp4, 5)
    md_text = insert_before_heading(md_text, "## 15. 스터디에서 설명하고 함께 점검하기", cp5, 5)
    return md_text


def inject_week06_active_learning(md_text: str) -> str:
    cp1 = """
<MicroCheckpoint
  :week="6"
  checkpointId="cp1-check"
  mode="choice"
  badge="CHECKPOINT 1 · 3~5절 오개념 함정 진단 (O/X)"
  title="Triton으로 직접 짠 Vector Add 커널은 PyTorch 기본 x + y보다 2배 이상 빠를까?"
  question="Triton으로 첫 블록 커널인 add_kernel(Z = X + Y)을 완벽하게 작성했습니다. 충분히 큰 벡터(N = 10⁸)에서 PyTorch 기본 연산(x + y)과 실행 속도를 비교하면 어떤 결과가 나올까요?"
  :options="[
    {
      label: 'Triton 커널이 저수준 포인터를 직접 제어하므로 PyTorch 기본 연산보다 2배 이상 빠르게 측정된다.',
      correct: false,
      feedback: '5절의 결론을 확인해 보세요! PyTorch 기본 x + y 역시 내부적으로 이미 잘 최적화된 컴파일된 GPU 커널을 호출합니다.'
    },
    {
      label: '두 방식 모두 X 읽기(1N) + Y 읽기(1N) + Z 쓰기(1N) = 총 3N 원소를 HBM과 주고받는 최소 이동량에 이미 도달해 있어 속도가 거의 비슷하다.',
      correct: true,
      feedback: '정확합니다! 단일 연산(x + y)에는 제거할 중간 텐서가 없으므로 둘 다 똑같이 HBM 대역폭 한계선에서 실행됩니다. Triton의 진가는 여러 단계 연산이 이어지는 다음 6절의 Softmax에서 폭발합니다!'
    }
  ]"
/>

"""
    cp2 = """
<MicroCheckpoint
  :week="6"
  checkpointId="cp2-predict"
  mode="choice"
  badge="CHECKPOINT 2 · 7절 읽기 전 버그 예측 (Spot-the-Bug)"
  title="7절을 읽기 전에 찾아보세요: 마스킹 패딩값에 other=0.0을 넣으면 왜 Softmax가 틀릴까요?"
  question="길이 N = 5인 행 [-5.0, -2.0, -8.0, -3.0, -4.0]을 BLOCK_SIZE = 8 블록 하나에 읽으면서 row = tl.load(ptrs, mask=offs < N, other=0.0) 으로 작성한 뒤 곧바로 tl.max(row, axis=0) 와 tl.exp(row - row_max) 를 계산했습니다. 어떤 심각한 버그가 발생할까요?"
  :options="[
    {
      label: '아무 문제 없다 — 0.0은 덧셈의 항등원이므로 Softmax 확률 결과에 영향을 주지 않는다.',
      correct: false,
      feedback: '0.0은 덧셈에서는 항등원이지만, 최댓값(max) 비교와 지수 함수(exp)에서는 절대 투명한 값이 아닙니다!'
    },
    {
      label: '유효 원소가 모두 음수일 때 패딩된 0.0이 가짜 최댓값(row_max = 0.0)이 되어버리고, exp(0.0) = 1.0이 분모 합(sum)에 더해져 확률값이 완전히 왜곡된다.',
      correct: true,
      feedback: '정답입니다! 최댓값에 영향을 주지 않으면서 동시에 exp(other - max) = 0 이 되어 분모 합에서도 자동으로 0이 되게 하려면 반드시 other = -float(&quot;inf&quot;) 로 채워야 합니다. 바로 아래 7절에서 확인해 보세요!'
    }
  ]"
/>

"""
    cp3 = """
<MicroCheckpoint
  :week="6"
  checkpointId="cp3-calc"
  mode="number"
  badge="CHECKPOINT 3 · 6~8절 Fused Softmax 데이터 이동 감소 배수"
  title="8MN에서 2MN으로! Triton Fused Softmax는 HBM 이동량을 몇 배 줄였을까?"
  question="크기 [M, N] 행렬에 대해 PyTorch 기본 연산을 나열한 naive_softmax는 주요 큰 텐서 기준으로 읽기 5MN + 쓰기 3MN = 총 8MN개의 원소를 HBM과 주고받았습니다. 반면 한 행을 온칩 SRAM에 한 번만 읽어와(1MN) 최댓값·지수·합·나눗셈을 모두 마치고 최종 결과만 기록하는(1MN) Triton Fused Softmax는 총 2MN개만 이동합니다. 주요 HBM 원소 이동량은 몇 배(8MN ÷ 2MN) 줄어들었나요?"
  :expectedAnswer="4"
  :tolerance="0.01"
  unit="배"
  placeholder="정수 입력"
  hint="분리 실행의 총 원소 이동량(8MN)을 Fused 커널의 총 원소 이동량(2MN)으로 나누어 보세요!"
  solutionExplanation="정확합니다! 8MN ÷ 2MN = 4배 감소합니다(이동량 75% 절감). 대역폭 병목 연산인 Softmax에서 HBM 왕복을 1/4로 압축했기 때문에 RTX 4070 등 실제 GPU 실측에서도 수 배의 속도 향상이 그대로 나타납니다!"
/>

"""
    cp4 = """
## 🧩 실습 검증: Triton 블록 커널 핵심 구문 매칭 미션 (Checkpoint 4)

<ConceptMatchQuest
  :week="6"
  checkpointId="cp4-quest"
  badge="CHECKPOINT 4 · Triton 핵심 API & 동작 매칭 미션"
  title="Fused Softmax 커널을 구성하는 4가지 Triton 구문과 역할을 연결해 보세요!"
  instruction="왼쪽의 Triton 코드 조각을 먼저 클릭한 뒤, 오른쪽에서 그 코드가 담당하는 정확한 실행 의미를 골라 4쌍을 완성하세요."
  :pairs="[
    {
      id: 'pid',
      left: 'row_idx = tl.program_id(0)',
      right: '현재 실행 중인 커널 프로그램 인스턴스가 행렬의 몇 번째 행(Row)을 맡았는지 식별'
    },
    {
      id: 'arange',
      left: 'col_offsets = tl.arange(0, BLOCK_SIZE)',
      right: '단일 스칼라 인덱스가 아니라 한 행 안의 열 오프셋 벡터 [0, 1, ..., BLOCK_SIZE-1]을 한 번에 생성'
    },
    {
      id: 'mask',
      left: 'tl.load(ptrs, mask=col_offsets < N, other=-float(&quot;inf&quot;))',
      right: '열 길이 N을 넘는 패딩 구간을 메모리 오류 없이 막고, max/exp 리덕션 결과를 오염시키지 않게 처리'
    },
    {
      id: 'reduce',
      left: 'tl.max(row, axis=0) & tl.sum(num, axis=0)',
      right: 'HBM에 중간 결과를 내리지 않고 온칩 메모리(레지스터/SRAM) 안에서 행 단위 최댓값과 분모 합을 즉시 축약'
    }
  ]"
/>

<details class="t2s-solution-toggle">
<summary>🔍 미리 보기: 방금 작성한 Triton 커널이 7주차에서 GPU 기계어(PTX/SASS)로 컴파일되는 과정 펼쳐보기</summary>

<IrPipelineViewer />

</details>

---

"""
    cp5 = """
## 🏅 Week 06 완독 점검 & 인증 패스포트 (Checkpoint 5)

<FeynmanPassport
  :week="6"
  weekTitle="Python에서 Triton 커널로 — 블록 프로그래밍과 Fused Softmax"
  checkpointId="cp5-feynman"
  :allCheckpointIds="['cp1-check', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman']"
  promptQuestion="질문: Triton으로 Vector Add를 작성하면 PyTorch 기본 연산과 속도가 거의 비슷하지만, Softmax를 Triton 단일 커널로 작성하면 수 배 빨라지는 이유를 '온칩 메모리'와 'HBM 왕복' 관점에서 설명해 보세요."
  modelAnswer="Vector Add는 입력 두 개를 읽고 결과 하나를 쓰는 단일 연산이라 원래부터 중간 텐서의 HBM 왕복이 없어 더 줄일 데이터 이동이 없습니다. 반면 분리된 Softmax는 최댓값 빼기, exp, 합산, 나눗셈마다 중간 텐서를 HBM에 반복해서 썼다 읽지만(약 8MN), Triton Fused Softmax는 한 행을 온칩 SRAM/레지스터에 한 번만 읽어와 모든 단계를 끝내고 최종 결과만 HBM에 기록(2MN)하므로 HBM 이동량이 1/4로 줄어들어 크게 빨라집니다."
  :selfCheckItems="[
    'Vector Add와의 차이 언급: 단일 연산은 제거할 중간 텐서가 없지만 다단계 연산(Softmax)은 중간 HBM 왕복이 많다는 점을 대조했다.',
    '온칩 재사용 효과 언급: 한 행을 온칩 메모리(SRAM/레지스터)에 유지한 채 리덕션과 원소별 연산을 한 번에 끝내 HBM 읽기/쓰기를 줄인 점을 설명했다.'
  ]"
/>

---

"""
    md_text = insert_before_heading(md_text, "## 6. 커널 결합과 HBM 왕복 — Softmax에서 차이가 커지는 이유", cp1, 6)
    md_text = insert_before_heading(md_text, "## 7. 행 단위 리덕션과 패딩값 — 한 행을 블록 하나에 담기", cp2, 6)
    md_text = insert_before_heading(md_text, "## 8. 공동 활동 — 블록 인덱싱과 Fused Softmax의 데이터 이동 비교", cp3, 6)
    md_text = insert_before_heading(md_text, "## 9. 정확도 검증과 벤치마크 — 무엇을 확인하고 어느 시간을 잴까", cp4, 6)
    md_text = insert_before_heading(md_text, "## 15. 스터디에서 설명하고 함께 점검하기", cp5, 6)
    return md_text


def inject_week07_active_learning(md_text: str) -> str:
    cp1 = """
<MicroCheckpoint
  :week="7"
  checkpointId="cp1-check"
  mode="choice"
  badge="CHECKPOINT 1 · 3~4절 오개념 함정 진단 (O/X)"
  title="Grouped Ordering(GROUP_SIZE_M)은 어떻게 타일 크기를 안 바꾸고도 성능을 높일까?"
  question="4절에서 출력 타일 크기(BM, BN)는 그대로 둔 채, 프로그램 ID를 2차원 타일 좌표(pid_m, pid_n)로 매핑하는 순서만 Row-major에서 Grouped Ordering으로 바꿨습니다. 이 기법이 성능을 높이는 진짜 원리는 무엇일까요?"
  :options="[
    {
      label: '프로그램 실행 순서를 묶어 주면 일부 타일의 내적 곱셈을 생략할 수 있어 총 FLOPs가 줄어든다.',
      correct: false,
      feedback: '실행 순서를 어떻게 바꾸든 모든 출력 타일 C[pid_m, pid_n]을 구하기 위한 총 연산량(2MNK)과 각 블록의 계산량은 100% 동일합니다!'
    },
    {
      label: '동시에 실행되는 프로그램 블록들이 인접한 M·N 영역에 모이게 되어, 같은 A 행 타일과 B 열 타일을 칩 공유 L2 캐시에서 함께 재사용함으로써 HBM 실효 읽기 트래픽을 줄인다.',
      correct: true,
      feedback: '정확합니다! 한 행을 끝까지 도는 Row-major 순서에서는 B의 열 타일 종류가 너무 많아져 L2 캐시에서 밀려나지만, GROUP_SIZE_M 단위로 묶어 돌면 적은 수의 A·B 타일을 여러 SM이 L2 캐시 위에서 높은 적중률로 공유할 수 있습니다.'
    }
  ]"
/>

"""
    cp2 = """
<MicroCheckpoint
  :week="7"
  checkpointId="cp2-predict"
  mode="choice"
  badge="CHECKPOINT 2 · 6절 읽기 전 결과 예측 (Predict-Before-Read)"
  title="6절을 읽기 전에 예측해 보세요: 출력 타일(128×128)은 고정하고 BK만 32 → 64로 키우면?"
  question="4096×4096 FP16 행렬곱에서 출력 타일 크기(BM = 128, BN = 128)는 그대로 두고 K축 타일 폭(BLOCK_SIZE_K)만 32에서 64로 2배 키웠습니다. 캐시 미포함 이상적 타일 모델에서 ① 전체 HBM 입력 읽기량(Q_read)과 ② 1단계 Shared Memory 타일 크기는 각각 어떻게 변할까요?"
  :options="[
    {
      label: 'HBM 입력 읽기량(Q_read)이 절반으로 줄어들고, 1단계 Shared Memory 크기는 그대로 유지된다.',
      correct: false,
      feedback: '정반대입니다! BK가 2배 커지면 K 루프 반복 횟수는 절반이 되지만 한 번에 읽는 조각 폭이 2배가 되어 블록당 총 읽는 면적(BM×K + K×BN)은 변하지 않습니다.'
    },
    {
      label: '이상적 HBM 입력 읽기량(Q_read)은 그대로 유지되고, 한 번에 온칩에 올리는 A·B 타일 폭이 넓어져 1단계 Shared Memory 요구량이 2배(16 KiB → 32 KiB)로 커진다.',
      correct: true,
      feedback: '완벽합니다! 산술 강도의 큰 틀은 출력 타일(BM, BN)이 결정하고, BK와 num_stages는 한 번의 루프 입도와 온칩 Shared Memory 파이프라인 용량을 결정합니다. 바로 아래 6절 계산표에서 확인해 보세요!'
    }
  ]"
/>

"""
    cp3 = """
<MicroCheckpoint
  :week="7"
  checkpointId="cp3-calc"
  mode="number"
  badge="CHECKPOINT 3 · 6절 다중 스테이지 Shared Memory 용량 계산"
  title="후보 2(128×128, BK=32, FP16)에서 num_stages=4일 때 파이프라인 버퍼 크기는 몇 KiB일까?"
  question="6.2절 후보 2(BM = 128, BN = 128, BK = 32, FP16 = 2 Byte)에서 1단계 입력 타일 크기는 A 타일(128×32×2B = 8 KiB) + B 타일(32×128×2B = 8 KiB) = 총 16 KiB였습니다. 비동기 복사와 연산을 겹치기 위해 소프트웨어 파이프라이닝 단계 수를 num_stages = 4로 설정할 때(교안 식 S_pipe = num_stages × S_1stage 기준), 필요한 파이프라인 타일 버퍼 용량은 몇 KiB인가요?"
  :expectedAnswer="64"
  :tolerance="0.1"
  unit="KiB"
  placeholder="정수 입력"
  hint="1단계 A+B 합계 용량(16 KiB)에 스테이지 수(num_stages = 4)를 곱해 보세요: 4 × 16 KiB !"
  solutionExplanation="정확합니다! 4 × 16 KiB = 64 KiB입니다. 교육용 SM 한도(96 KiB) 안에 여유 있게 들어오므로 컴파일과 실행이 가능하지만, 만약 같은 128×128에서 BK=64, num_stages=4로 잡으면 4 × 32 = 128 KiB > 96 KiB가 되어 OutOfResources 오류가 발생합니다!"
/>

"""
    cp4 = """
## 🎛️ 실습 검증: 다단계 IR 파이프라인 탐정 미션 & 시뮬레이터 (Checkpoint 4)

> 방금 7~8절에서 배운 Triton 컴파일 파이프라인(`Python` → `TTIR` → `TTGIR` → `LLVM IR` → `PTX` → `SASS`)의 실제 산출물을 아래 탐색기에서 직접 클릭해 비교하고 **탐정 미션**을 통과해 보세요! 이어지는 Roofline 계산기에서는 A100·RTX 5090 프리셋도 자유롭게 실험할 수 있습니다.

<IrPipelineViewer :week="7" checkpointId="cp4-quest" />

<RooflineCalculator />

---

"""
    cp5 = """
## 🏅 Week 07 완독 점검 & 인증 패스포트 (Checkpoint 5)

<FeynmanPassport
  :week="7"
  weekTitle="Tiled MatMul과 컴파일 과정 — 2D 타일 재사용에서 GPU 기계어까지"
  checkpointId="cp5-feynman"
  :allCheckpointIds="['cp1-check', 'cp2-predict', 'cp3-calc', 'cp4-quest', 'cp5-feynman']"
  promptQuestion="질문: 프로그래머가 Python으로 작성한 단일 @triton.jit 행렬곱 커널이 코드 수정 없이도 A100(sm_80)과 RTX 5090(sm_120) 양쪽에서 높은 성능의 Tensor Core 기계어로 실행될 수 있는 이유를 'Autotune'과 '다단계 IR(TTIR → TTGIR)' 관점에서 설명해 보세요."
  modelAnswer="프로그래머는 블록 수준 알고리즘(포인터 전진과 tl.dot)만 작성해 하드웨어 독립적인 TTIR로 표현하고, 칩마다 다른 최적 타일 크기와 파이프라인 단계(BM, BN, BK, num_stages, num_warps)는 @triton.autotune이 실측으로 선택합니다. 이후 컴파일러 백엔드가 타겟 아키텍처(sm_80, sm_120)에 맞춰 TTGIR 단계에서 워프 분할(#mma)·Shared Memory 스위즐·비동기 복사를 자동 삽입하고 PTX/SASS 기계어로 하향 변환하기 때문입니다."
  :selfCheckItems="[
    '알고리즘과 튜닝 분리 언급: 커널 코드는 블록 단위로 유지하고 칩별 최적 타일/워프/스테이지 설정은 @triton.autotune으로 탐색한다는 점을 설명했다.',
    '다단계 IR 하향 변환 언급: 하드웨어 독립적 TTIR에서 타겟 GPU 종속 레이아웃(#mma, #shared)과 비동기 파이프라인을 입히는 TTGIR → PTX/SASS 변환 역할을 짚었다.'
  ]"
/>

---

"""
    md_text = insert_before_heading(md_text, "## 5. 메타파라미터와 자동 튜닝", cp1, 7)
    md_text = insert_before_heading(md_text, "## 6. 계산 예제 2", cp2, 7)
    md_text = insert_before_heading(md_text, "## 7. Triton 컴파일 파이프라인과 IR 계층", cp3, 7)
    md_text = insert_before_heading(md_text, "## 9. 공동 활동 정리", cp4, 7)
    md_text = insert_before_heading(md_text, "## 14. 스터디에서 설명하고 함께 점검하기", cp5, 7)
    return md_text


def inject_active_learning(week_num: int, md_text: str) -> str:
    if week_num == 2:
        return inject_week02_active_learning(md_text)
    elif week_num == 3:
        return inject_week03_active_learning(md_text)
    elif week_num == 4:
        return inject_week04_active_learning(md_text)
    elif week_num == 5:
        return inject_week05_active_learning(md_text)
    elif week_num == 6:
        return inject_week06_active_learning(md_text)
    elif week_num == 7:
        return inject_week07_active_learning(md_text)
    return md_text


def generate_all():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "intro").mkdir(parents=True, exist_ok=True)

    # 1. Build Home Page (docs/index.md)
    index_md = """---
layout: home

hero:
  name: "Tensor2Silicon LAB"
  text: "AI 모델에서 가속기 커널까지"
  tagline: "PyTorch·Transformer 수식 한 줄이 그래프(IR), 컴파일러, 런타임을 거쳐 GPU·TPU·NPU 실리콘 위에서 실행되기까지의 전 과정을 정량적 손계산·능동 학습 퀘스트·커널 실습으로 체득하는 오픈 리서치 웹북"
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
  - icon: 🎯
    title: "전 주차 5단계 능동 학습 & 완독 패스포트"
    details: "단순히 눈으로만 읽고 지나치지 않도록 매 주차 3분 직관 지도, 오개념 O/X, 손계산 빈칸, 인터랙티브 실습 미션, 파인만 한줄설명 인증서를 제공합니다."
  - icon: 🧮
    title: "정량적 손계산 & 인터랙티브 시뮬레이터"
    details: "FLOPs, HBM 메모리 이동량(Q), 산술 강도(I = F/Q), Shared Memory 타일 버퍼 크기를 손으로 직접 계산하고 웹북 내장 Roofline·IR 탐색기로 즉시 실험합니다."
  - icon: ⚙️
    title: "다단계 컴파일러 & IR 실측 해부"
    details: "PyTorch(torch.compile), JAX(OpenXLA), Modular MAX, 그리고 Triton 커널이 TTIR → TTGIR → LLVM IR → PTX → SASS로 구체화되는 과정을 A100 및 RTX 5090 실측 데이터로 검증합니다."
  - icon: 🧑‍🔬
    title: "방식 A: 메인 교안 + 6인 연구원 Deep-Dive"
    details: "체계적인 메인 교과서 흐름을 따라가면서도, 매 주차 6명의 연구원이 각자의 하드웨어와 시각에서 분석한 심화 리포트와 인터랙티브 슬라이드를 탭 한 번으로 함께 열람할 수 있습니다."
---

## 🗺️ 커리큘럼 한눈에 바로가기 (Curriculum Roadmap)

| 주차 | 주제 (Topic) | 능동 학습 퀘스트 & 인터랙티브 도구 (`5/5` 마스터리) | 바로가기 |
| :---: | :--- | :--- | :---: |
| **Intro** | **랩 소개 & 5대 실행 스택** | T2S Lab 운영 원칙, 5대 계층 아키텍처, 연구원 아카이브 매트릭스 | [소개 보기](./intro/overview) |
| **Week 02** | **모델에서 실리콘까지** | 실행 스택 역할 매칭 미션 + Epilogue Fusion 손계산 + **인터랙티브 HTML 슬라이드 덱** | [2주차 열기](./week02/) |
| **Week 03** | **커널의 시간은 어디에서 소비되는가** | 용량 vs 대역폭 진단 + Fusion 속도 배수 계산 + **병목 전환 타일 탐색 미션** | [3주차 열기](./week03/) |
| **Week 04** | **Roofline과 Tiling** | 산술 강도($I = F/Q$) 예측 + 타일 읽기량 손계산 + **Roofline & SRAM 제약 돌파 미션** | [4주차 열기](./week04/) |
| **Week 05** | **xPU를 어떻게 비교할 것인가** | 스펙시트 Sparsity 함정 진단 + 장치 역전 손계산 + **GPU·TPU & Prefill·Decode 매칭** | [5주차 열기](./week05/) |
| **Week 06** | **Python에서 Triton 커널로** | `-inf` 패딩 버그 찾기 + Fused Softmax `4x` 절감 계산 + **Triton 핵심 구문 매칭** | [6주차 열기](./week06/) |
| **Week 07** | **Tiled MatMul과 컴파일 과정** | L2 그룹핑 진단 + 파이프라인 SRAM 계산 + **다단계 IR(`TTIR→SASS`) 탐정 미션** | [7주차 열기](./week07/) |
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
        tldr_card_block = build_tldr_map_card(w)

        # 4a. Main Lecture Note (docs/week0X/index.md)
        main_raw = meta["main_src"].read_text(encoding="utf-8")
        # Ensure top title starts with # if it started with ##
        if main_raw.lstrip().startswith("## "):
            main_raw = main_raw.lstrip()[1:]

        main_processed = wrap_solutions_in_details(main_raw)
        main_processed = sanitize_markdown_for_vitepress(main_processed, week_num=w, is_researcher=False)
        main_processed = inject_active_learning(w, main_processed)

        # Collapse the long inline TOC (already shown on right sidebar) and place TldrMapCard right above it!
        main_processed, toc_subs = re.subn(
            r'\n## 목차\n(.*?)\n---\n',
            lambda m: (
                "\n"
                + tldr_card_block
                + '\n<details class="t2s-solution-toggle">\n'
                + '<summary>📑 전체 절 목록 펼쳐보기 (우측 고정 목차에서도 바로 이동 가능)</summary>\n\n'
                + m.group(1).strip()
                + "\n\n</details>\n\n---\n"
            ),
            main_processed,
            count=1,
            flags=re.DOTALL
        )
        if toc_subs != 1:
            raise ValueError(f"[Week {w}] Expected inline ## 목차 block for TldrMapCard placement, got {toc_subs}")

        # Insert synthesis_block right before the first `## 1.`
        insert_match = re.search(r'\n##\s+1\.', main_processed)
        if insert_match:
            pos = insert_match.start()
            main_final = (
                nav_bar
                + main_processed[:pos]
                + "\n"
                + synthesis_block
                + main_processed[pos:]
            )
        else:
            main_final = nav_bar + synthesis_block + main_processed

        (week_dir / "index.md").write_text(main_final, encoding="utf-8")

        # 4b. Guide & Reference Calculation Examples (docs/week0X/guide.md)
        guide_parts = [nav_bar]
        if meta["guide_src"] and meta["guide_src"].exists():
            g_raw = meta["guide_src"].read_text(encoding="utf-8")
            if g_raw.lstrip().startswith("## "):
                g_raw = g_raw.lstrip()[1:]
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
