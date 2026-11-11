import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const base = process.env.VITEPRESS_BASE || '/'

export default withMermaid(
  defineConfig({
    base,
    lang: 'ko-KR',
    title: 'Tensor2Silicon LAB',
    description: 'AI 모델에서 가속기 커널까지 — From Tensor to Silicon 오픈 리서치 웹북',
    ignoreDeadLinks: true,

    markdown: {
      math: true,
      lineNumbers: false,
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
        { text: '📖 랩 소개 & 아키텍처', link: '/intro/overview' },
        {
          text: '📚 주차별 커리큘럼',
          items: [
            { text: 'Week 02 · 모델에서 실리콘까지', link: '/week02/' },
            { text: 'Week 03 · 커널의 시간은 어디에서 소비되는가', link: '/week03/' },
            { text: 'Week 04 · Roofline과 Tiling', link: '/week04/' },
            { text: 'Week 05 · xPU를 어떻게 비교할 것인가', link: '/week05/' },
            { text: 'Week 06 · Python에서 Triton 커널로', link: '/week06/' },
            { text: 'Week 07 · Tiled MatMul과 컴파일 과정', link: '/week07/' }
          ]
        },
        { text: '✍️ 저자 & 연구원', link: '/intro/contributors' }
      ],

      sidebar: [
        {
          text: '🚀 시작하기 (Overview)',
          collapsed: false,
          items: [
            { text: '📖 T2S Lab 소개 & 5대 계층', link: '/intro/overview' },
            { text: '✍️ 대표 저자 & 연구원 아카이브', link: '/intro/contributors' }
          ]
        },
        {
          text: 'Week 02 · 모델에서 실리콘까지',
          collapsed: false,
          items: [
            { text: '📘 메인 교안 (실행 스택 해부)', link: '/week02/' },
            { text: '🧭 2주차 학습 안내', link: '/week02/guide' },
            {
              text: '🧑‍🔬 연구원 리서치 노트 (5인)',
              collapsed: true,
              items: [
                { text: 'mbsong · PyTorch·XLA·MAX 2부작', link: '/week02/researchers/mbsong' },
                { text: 'singon · 아키텍처 다이어그램 해부', link: '/week02/researchers/singon' },
                { text: 'junpark · 실행 스택 핵심 요약', link: '/week02/researchers/junpark' },
                { text: 'injokim · 인터랙티브 HTML & 덱', link: '/week02/researchers/injokim' },
                { text: 'jaehyeong · 컴파일 파이프라인 정리', link: '/week02/researchers/jaehyeong' }
              ]
            }
          ]
        },
        {
          text: 'Week 03 · 커널의 시간과 메모리',
          collapsed: false,
          items: [
            { text: '📘 메인 교안 (연산 vs 메모리 병목)', link: '/week03/' },
            { text: '🧭 3주차 안내 & 계산 예제', link: '/week03/guide' },
            {
              text: '🧑‍🔬 연구원 리서치 노트 (5인)',
              collapsed: true,
              items: [
                { text: 'mbsong · GPU·TPU 메모리 계층 비교', link: '/week03/researchers/mbsong' },
                { text: 'singon · 손계산 & 병목 판별', link: '/week03/researchers/singon' },
                { text: 'junpark · SRAM vs DRAM 물리 구조', link: '/week03/researchers/junpark' },
                { text: 'injokim · 수식 모델링 & 정량 분석', link: '/week03/researchers/injokim' },
                { text: 'jaehyeong · 메모리 계층 & 실전 예제', link: '/week03/researchers/jaehyeong' }
              ]
            }
          ]
        },
        {
          text: 'Week 04 · Roofline과 Tiling',
          collapsed: false,
          items: [
            { text: '📘 메인 교안 (산술 강도와 타일링)', link: '/week04/' },
            { text: '🧭 4주차 안내 & 계산 예제', link: '/week04/guide' },
            {
              text: '🧑‍🔬 연구원 리서치 노트 (5인)',
              collapsed: true,
              items: [
                { text: 'mbsong · 타일 크기별 Roofline 분석', link: '/week04/researchers/mbsong' },
                { text: 'singon · 4×4 & 4096 손계산 해설', link: '/week04/researchers/singon' },
                { text: 'junpark · 2D Block Tiling 직관', link: '/week04/researchers/junpark' },
                { text: 'injokim · 수식 유도 & 연습문제 풀이', link: '/week04/researchers/injokim' },
                { text: 'jaehyeong · Roofline & Tiling 요약', link: '/week04/researchers/jaehyeong' }
              ]
            }
          ]
        },
        {
          text: 'Week 05 · xPU를 어떻게 비교할 것인가',
          collapsed: false,
          items: [
            { text: '📘 메인 교안 (GPU·TPU·NPU 비교)', link: '/week05/' },
            { text: '🧭 5주차 안내 & 비교 계산 예제', link: '/week05/guide' },
            {
              text: '🧑‍🔬 연구원 리서치 노트 (3인)',
              collapsed: true,
              items: [
                { text: 'mbsong · GPU vs TPU vs NPU 비교', link: '/week05/researchers/mbsong' },
                { text: 'singon · xPU 비교 기준 & 손계산', link: '/week05/researchers/singon' },
                { text: 'junpark · 아키텍처 심층 리서치', link: '/week05/researchers/junpark' }
              ]
            }
          ]
        },
        {
          text: 'Week 06 · Python에서 Triton 커널로',
          collapsed: false,
          items: [
            { text: '📘 메인 교안 (1D 블록 & Softmax)', link: '/week06/' },
            { text: '🧭 6주차 학습 안내', link: '/week06/guide' },
            {
              text: '🧑‍🔬 연구원 리서치 노트 (5인)',
              collapsed: true,
              items: [
                { text: 'mbsong · Vector Add & Fused Softmax', link: '/week06/researchers/mbsong' },
                { text: 'singon · RTX 4070 실측 벤치마크', link: '/week06/researchers/singon' },
                { text: 'junpark · Online Softmax & IR 분석', link: '/week06/researchers/junpark' },
                { text: 'injokim · 블록 포인터 & 마스킹 해부', link: '/week06/researchers/kiminjo' },
                { text: 'jaehyeong · Triton 핵심 요약', link: '/week06/researchers/jaehyeong' }
              ]
            }
          ]
        },
        {
          text: 'Week 07 · Tiled MatMul과 컴파일 과정',
          collapsed: false,
          items: [
            { text: '📘 메인 교안 (2D 타일 & 다단계 IR)', link: '/week07/' },
            { text: '🧭 7주차 안내 & 계산 예제', link: '/week07/guide' },
            {
              text: '🧑‍🔬 연구원 리서치 노트 (5인)',
              collapsed: true,
              items: [
                { text: 'mbsong · NVIDIA A100 IR·SASS 실측', link: '/week07/researchers/mbsong' },
                { text: 'junpark · RTX 5090 (sm_120) 실측 해부', link: '/week07/researchers/junpark' },
                { text: 'singon · L2 그룹핑 & IR 1:1 대응표', link: '/week07/researchers/singon' },
                { text: 'injokim · 알고리즘 & MLIR 하향 변환', link: '/week07/researchers/injokim' },
                { text: 'jaehyeong · 핵심 코드 & 컴파일 요약', link: '/week07/researchers/jaehyeong' }
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
        label: '📑 이 페이지 목차'
      },

      docFooter: {
        prev: '이전 장',
        next: '다음 장'
      },

      footer: {
        message: '메인 교안 대표 집필 & 커리큘럼 설계: 이영빈 (Youngbin Lee, @yblee110) · Supported by MSIT & IITP (RS-2026-25521385)',
        copyright: 'Copyright © 2026 Youngbin Lee (@yblee110), Legato User Group & Tensor2Silicon LAB Contributors (Apache-2.0)'
      }
    },

    mermaid: {
      theme: 'default'
    }
  })
)
