import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const base = process.env.VITEPRESS_BASE || '/'

export default withMermaid(
  defineConfig({
    base,
    lang: 'ko-KR',
    title: 'Tensor2Silicon',
    description: 'AI 모델에서 가속기 커널까지 — From Tensor to Silicon 기술 웹북 (집필: 이영빈)',
    ignoreDeadLinks: true,

    markdown: {
      math: true,
      lineNumbers: false,
      anchor: {
        slugify: (str: string) =>
          str
            .trim()
            .toLowerCase()
            .replace(/[^\p{L}\p{N}\s-]/gu, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-')
      },
      languageAlias: {
        cuda: 'cpp',
        mlir: 'cpp',
        llvm: 'asm',
        ptx: 'asm',
        sass: 'asm'
      }
    },

    themeConfig: {
      nav: [
        { text: '🏠 홈', link: '/' },
        { text: '📖 책 소개 & 아키텍처', link: '/intro/overview' },
        {
          text: '📚 전체 목차 (1장 ~ 6장)',
          items: [
            { text: 'Chapter 01 · 모델에서 실리콘까지', link: '/ch01/' },
            { text: 'Chapter 02 · 커널의 시간은 어디에서 소비되는가', link: '/ch02/' },
            { text: 'Chapter 03 · Roofline과 Tiling', link: '/ch03/' },
            { text: 'Chapter 04 · xPU를 어떻게 비교할 것인가', link: '/ch04/' },
            { text: 'Chapter 05 · Python에서 Triton 커널로', link: '/ch05/' },
            { text: 'Chapter 06 · Tiled MatMul과 컴파일 과정', link: '/ch06/' }
          ]
        },
        { text: '✍️ 저자 & 연구진', link: '/intro/contributors' }
      ],

      sidebar: [
        {
          text: '🚀 서문 & 아키텍처 (Preface)',
          collapsed: false,
          items: [
            { text: '📖 책 소개 & 5대 실행 스택', link: '/intro/overview' },
            { text: '✍️ 대표 저자 & 연구진 소개', link: '/intro/contributors' }
          ]
        },
        {
          text: 'Chapter 01 · 모델에서 실리콘까지',
          collapsed: false,
          items: [
            { text: '📘 전체 심화 본문 (실행 스택 해부)', link: '/ch01/' },
            { text: '⚡ 15분 핵심 압축 교안 (Core Digest)', link: '/ch01/guide' },
            {
              text: '🧑‍🔬 연구진 심화 노트 (5편)',
              collapsed: true,
              items: [
                { text: 'mbsong · PyTorch·XLA·MAX 2부작', link: '/ch01/researchers/mbsong' },
                { text: 'singon · 아키텍처 다이어그램 해부', link: '/ch01/researchers/singon' },
                { text: 'junpark · 실행 스택 핵심 요약', link: '/ch01/researchers/junpark' },
                { text: 'injokim · 인터랙티브 HTML & 덱', link: '/ch01/researchers/injokim' },
                { text: 'jaehyeong · 컴파일 파이프라인 정리', link: '/ch01/researchers/jaehyeong' }
              ]
            }
          ]
        },
        {
          text: 'Chapter 02 · 커널의 시간과 메모리',
          collapsed: false,
          items: [
            { text: '📘 전체 심화 본문 (연산 vs 메모리 병목)', link: '/ch02/' },
            { text: '⚡ 15분 핵심 압축 교안 (Core Digest)', link: '/ch02/guide' },
            {
              text: '🧑‍🔬 연구진 심화 노트 (5편)',
              collapsed: true,
              items: [
                { text: 'mbsong · GPU·TPU 메모리 계층 비교', link: '/ch02/researchers/mbsong' },
                { text: 'singon · 손계산 & 병목 판별', link: '/ch02/researchers/singon' },
                { text: 'junpark · SRAM vs DRAM 물리 구조', link: '/ch02/researchers/junpark' },
                { text: 'injokim · 수식 모델링 & 정량 분석', link: '/ch02/researchers/injokim' },
                { text: 'jaehyeong · 메모리 계층 & 실전 예제', link: '/ch02/researchers/jaehyeong' }
              ]
            }
          ]
        },
        {
          text: 'Chapter 03 · Roofline과 Tiling',
          collapsed: false,
          items: [
            { text: '📘 전체 심화 본문 (산술 강도와 타일링)', link: '/ch03/' },
            { text: '⚡ 15분 핵심 압축 교안 (Core Digest)', link: '/ch03/guide' },
            {
              text: '🧑‍🔬 연구진 심화 노트 (5편)',
              collapsed: true,
              items: [
                { text: 'mbsong · 타일 크기별 Roofline 분석', link: '/ch03/researchers/mbsong' },
                { text: 'singon · 4×4 & 4096 손계산 해설', link: '/ch03/researchers/singon' },
                { text: 'junpark · 2D Block Tiling 직관', link: '/ch03/researchers/junpark' },
                { text: 'injokim · 수식 유도 & 연습문제 풀이', link: '/ch03/researchers/injokim' },
                { text: 'jaehyeong · Roofline & Tiling 요약', link: '/ch03/researchers/jaehyeong' }
              ]
            }
          ]
        },
        {
          text: 'Chapter 04 · xPU를 어떻게 비교할 것인가',
          collapsed: false,
          items: [
            { text: '📘 전체 심화 본문 (GPU·TPU·NPU 비교)', link: '/ch04/' },
            { text: '⚡ 15분 핵심 압축 교안 (Core Digest)', link: '/ch04/guide' },
            {
              text: '🧑‍🔬 연구진 심화 노트 (3편)',
              collapsed: true,
              items: [
                { text: 'mbsong · GPU vs TPU vs NPU 비교', link: '/ch04/researchers/mbsong' },
                { text: 'singon · xPU 비교 기준 & 손계산', link: '/ch04/researchers/singon' },
                { text: 'junpark · 아키텍처 심층 리서치', link: '/ch04/researchers/junpark' }
              ]
            }
          ]
        },
        {
          text: 'Chapter 05 · Python에서 Triton 커널로',
          collapsed: false,
          items: [
            { text: '📘 전체 심화 본문 (1D 블록 & Softmax)', link: '/ch05/' },
            { text: '⚡ 15분 핵심 압축 교안 (Core Digest)', link: '/ch05/guide' },
            {
              text: '🧑‍🔬 연구진 심화 노트 (5편)',
              collapsed: true,
              items: [
                { text: 'mbsong · Vector Add & Fused Softmax', link: '/ch05/researchers/mbsong' },
                { text: 'singon · RTX 4070 실측 벤치마크', link: '/ch05/researchers/singon' },
                { text: 'junpark · Online Softmax & IR 분석', link: '/ch05/researchers/junpark' },
                { text: 'injokim · 블록 포인터 & 마스킹 해부', link: '/ch05/researchers/kiminjo' },
                { text: 'jaehyeong · Triton 핵심 요약', link: '/ch05/researchers/jaehyeong' }
              ]
            }
          ]
        },
        {
          text: 'Chapter 06 · Tiled MatMul과 컴파일 과정',
          collapsed: false,
          items: [
            { text: '📘 전체 심화 본문 (2D 타일 & 다단계 IR)', link: '/ch06/' },
            { text: '⚡ 15분 핵심 압축 교안 (Core Digest)', link: '/ch06/guide' },
            {
              text: '🧑‍🔬 연구진 심화 노트 (5편)',
              collapsed: true,
              items: [
                { text: 'mbsong · NVIDIA A100 IR·SASS 실측', link: '/ch06/researchers/mbsong' },
                { text: 'junpark · RTX 5090 (sm_120) 실측 해부', link: '/ch06/researchers/junpark' },
                { text: 'singon · L2 그룹핑 & IR 1:1 대응표', link: '/ch06/researchers/singon' },
                { text: 'injokim · 알고리즘 & MLIR 하향 변환', link: '/ch06/researchers/injokim' },
                { text: 'jaehyeong · 핵심 코드 & 컴파일 요약', link: '/ch06/researchers/jaehyeong' }
              ]
            }
          ]
        }
      ],

      socialLinks: [
        { icon: 'github', link: 'https://github.com/legato-user-group/tensor2silicon-lab' }
      ],

      search: {
        provider: 'local',
        options: {
          translations: {
            button: {
              buttonText: '웹북 검색 (수식·용어·코드)',
              buttonAriaLabel: '웹북 검색'
            },
            modal: {
              noResultsText: '검색 결과가 없습니다',
              resetButtonTitle: '초기화',
              footer: {
                selectText: '선택',
                navigateText: '이동',
                closeText: '닫기'
              }
            }
          }
        }
      },

      outline: {
        level: [2, 3],
        label: '📑 이 장의 목차'
      },

      docFooter: {
        prev: '이전 장',
        next: '다음 장'
      },

      footer: {
        message: '대표 저자 & 커리큘럼 설계: 이영빈 (Youngbin Lee, @yblee110) · Supported by MSIT & IITP (RS-2026-25521385)',
        copyright: 'Copyright © 2026 Youngbin Lee (@yblee110), Legato User Group & Tensor2Silicon Contributors (Apache-2.0)'
      }
    },

    mermaid: {
      theme: 'default'
    }
  })
)
