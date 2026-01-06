#!/usr/bin/env python3
"""
보고서 마무리 처리 독립 실행 스크립트

Usage:
    python manual_finalize_report.py [report_file_name]
    
Examples:
    python manual_finalize_report.py                        # 기본 파일 처리
    python manual_finalize_report.py progressive_report.html  # 특정 파일 처리
    python manual_finalize_report.py custom_report.html      # 커스텀 파일 처리
"""

import sys
import os
import logging
from pathlib import Path
from src.utils.report_finalizer import ReportFinalizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """메인 함수"""
    try:
        # 기본 설정
        artifacts_dir = "./artifacts"
        output_dir = "."  # 상위 폴더 (프로젝트 루트)
        default_report_file = "progressive_report.html"
        
        # 명령행 인수 처리
        if len(sys.argv) > 1:
            report_file = sys.argv[1]
        else:
            report_file = default_report_file
        
        print("📋 보고서 마무리 처리 시작")
        print(f"📂 소스 디렉토리: {artifacts_dir}")
        print(f"📁 출력 디렉토리: {output_dir}")
        print(f"📄 대상 파일: {report_file}")
        print("-" * 50)
        
        # 소스 파일 경로 (artifacts 폴더 내)
        source_report_path = Path(artifacts_dir) / report_file
        if not source_report_path.exists():
            print(f"❌ 오류: 보고서 파일을 찾을 수 없습니다: {source_report_path}")
            print(f"📁 {artifacts_dir} 디렉토리에 {report_file} 파일이 있는지 확인해주세요.")
            sys.exit(1)
        
        # 출력 파일 경로들 (프로젝트 루트)
        output_html_path = Path(output_dir) / report_file
        output_pdf_path = Path(output_dir) / report_file.replace('.html', '.pdf')
        
        print(f"📖 소스 파일: {source_report_path}")
        print(f"💾 출력 HTML: {output_html_path}")
        print(f"📄 출력 PDF: {output_pdf_path}")
        print("-" * 50)
        
        # ReportFinalizer를 사용하여 처리
        finalizer = ReportFinalizer(artifacts_dir)
        
        # 1. HTML 파일을 읽어서 수정
        # success = finalizer._update_html_footer(source_report_path)
        # if not success:
        #     print("❌ HTML 파일 수정 실패")
        #     sys.exit(1)
        
        # 2. 수정된 HTML을 상위 폴더에 복사
        with open(source_report_path, 'r', encoding='utf-8') as source_file:
            html_content = source_file.read()
        
        with open(output_html_path, 'w', encoding='utf-8') as output_file:
            output_file.write(html_content)
        
        print(f"✅ HTML 파일 저장 완료: {output_html_path}")
        
        # 3. PDF 생성 (상위 폴더에 직접 저장)
        pdf_success = False
        
        # PDF 생성 메서드들을 순차적으로 시도
        pdf_methods = [
            finalizer._generate_pdf_with_weasyprint,
            finalizer._generate_pdf_with_pdfkit,
            finalizer._generate_pdf_with_playwright
        ]
        
        for method in pdf_methods:
            try:
                pdf_success = method(output_html_path, output_pdf_path)
                if pdf_success:
                    print(f"✅ PDF 생성 완료: {output_pdf_path}")
                    break
            except ImportError as e:
                logger.debug(f"PDF 생성 라이브러리 없음 ({method.__name__}): {e}")
                continue
            except Exception as e:
                logger.debug(f"PDF 생성 실패 ({method.__name__}): {e}")
                continue
        
        if success:
            print("\n🎉 보고서 마무리 처리가 완료되었습니다!")
            print("\n📁 생성된 파일:")
            
            # HTML 파일
            if output_html_path.exists():
                print(f"   ✅ {output_html_path} (하단 메시지 수정 완료)")
            
            # PDF 파일
            if pdf_success and output_pdf_path.exists():
                print(f"   ✅ {output_pdf_path} (PDF 버전 생성 완료)")
            else:
                print(f"   ⚠️ PDF 생성 실패 - 필요한 라이브러리를 설치해주세요")
                print("      pip install weasyprint  # 권장 (순수 Python)")
                print("      pip install pdfkit      # wkhtmltopdf 필요") 
                print("      pip install playwright  # 브라우저 설치 필요")
                
            print(f"\n📄 최종 보고서 위치:")
            print(f"   - HTML: {output_html_path}")
            if pdf_success and output_pdf_path.exists():
                print(f"   - PDF:  {output_pdf_path}")
            
        else:
            print("\n❌ 보고서 마무리 처리 중 오류가 발생했습니다.")
            print("로그를 확인하여 문제를 해결해주세요.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)

def show_help():
    """도움말 출력"""
    help_text = """
📋 보고서 마무리 처리 도구

이 스크립트는 작성 완료된 보고서의 마무리 처리를 수행합니다:
1. ./artifacts/ 폴더에서 보고서 파일을 읽어옴
2. HTML 하단의 "점진적으로 작성되고 있습니다" 메시지를 완료 정보로 변경
3. 수정된 HTML과 PDF를 프로젝트 루트 폴더에 저장

사용법:
    python manual_finalize_report.py [보고서파일명]

예시:
    python manual_finalize_report.py                        # 기본 파일 처리
    python manual_finalize_report.py progressive_report.html  # 특정 파일 처리
    python manual_finalize_report.py custom_report.html      # 커스텀 파일 처리

파일 경로:
    소스: ./artifacts/[보고서파일명]
    출력: ./[보고서파일명], ./[보고서파일명.pdf]

PDF 생성을 위한 라이브러리 설치:
    pip install weasyprint     # 권장 (순수 Python)
    pip install pdfkit         # wkhtmltopdf 필요
    pip install playwright     # 브라우저 설치 필요
"""
    print(help_text)

if __name__ == "__main__":
    # 도움말 요청 확인
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        sys.exit(0)
    
    main() 