from typing import Dict, Any, List
from src.tools.python_repl import handle_python_repl_tool
from src.tools.bash_tool import handle_bash_tool

tool_list = [
    {
        "toolSpec": {
            "name": "python_repl_tool",
            "description": "Use this to execute python code and do data analysis or calculation. If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The python code to execute to do further analysis or calculation."
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "bash_tool",
            "description": "Use this to execute bash command and do necessary operations.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "The bash command to be executed."
                        }
                    },
                    "required": ["cmd"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "generate_html_report",
            "description": "Generate a comprehensive HTML report with embedded charts and styling. Use this tool to create the final HTML report. CRITICAL: All references must be verified against ./artifacts/research_data.txt - do not include any unverified citations.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the report"
                        },
                        "content": {
                            "type": "string",
                            "description": "The main content of the report in HTML format. Ensure any references are verified in ./artifacts/research_data.txt"
                        },
                        "include_charts": {
                            "type": "boolean",
                            "description": "Whether to include all available charts from artifacts folder",
                            "default": True
                        }
                    },
                    "required": ["title", "content"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "generate_markdown_report",
            "description": "Generate a comprehensive Markdown report with image references. Use this tool to create the final Markdown report. CRITICAL: All references must be verified against ./artifacts/research_data.txt - do not include any unverified citations.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the report"
                        },
                        "content": {
                            "type": "string",
                            "description": "The main content of the report in Markdown format. Ensure any references are verified in ./artifacts/research_data.txt"
                        },
                        "include_images": {
                            "type": "boolean",
                            "description": "Whether to include all available images from artifacts folder",
                            "default": True
                        }
                    },
                    "required": ["title", "content"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "generate_pdf_report",
            "description": "Generate a PDF report from HTML with proper Korean font support and professional formatting. This tool fixes formatting issues and ensures proper layout.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the report"
                        },
                        "content": {
                            "type": "string",
                            "description": "The main content of the report in HTML format"
                        },
                        "include_charts": {
                            "type": "boolean",
                            "description": "Whether to include all available charts from artifacts folder",
                            "default": True
                        },
                        "page_size": {
                            "type": "string",
                            "description": "Page size for the PDF (A4, Letter, etc.)",
                            "default": "A4"
                        }
                    },
                    "required": ["title", "content"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "estimate_bedrock_costs",
            "description": "Estimate Bedrock costs for agentic program execution and save to artifacts/bedrock_billing.txt. Calculates costs based on agent usage patterns and token consumption.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query_complexity": {
                            "type": "string",
                            "description": "Complexity level of the query (simple, medium, complex, very_complex)",
                            "enum": ["simple", "medium", "complex", "very_complex"],
                            "default": "medium"
                        },
                        "execution_count": {
                            "type": "integer",
                            "description": "Number of times the workflow will be executed",
                            "default": 1
                        },
                        "include_analysis": {
                            "type": "boolean",
                            "description": "Whether to include detailed cost breakdown analysis",
                            "default": True
                        }
                    },
                    "required": []
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "initialize_progressive_report",
            "description": "Initialize a new progressive report with basic structure. Creates or overwrites the base report file with outline structure.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The title of the report"
                        },
                        "format": {
                            "type": "string",
                            "description": "Report format (html or markdown)",
                            "enum": ["html", "markdown"],
                            "default": "html"
                        },
                        "sections": {
                            "type": "array",
                            "description": "List of main sections to include in the report outline",
                            "items": {"type": "string"},
                            "default": ["Executive Summary", "Analysis Results", "Key Findings", "Recommendations", "Conclusion"]
                        }
                    },
                    "required": ["title"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "add_section_to_report",
            "description": "Add or update a specific section in the progressive report. This allows building the report incrementally. CRITICAL: When adding '참고문헌' (References) sections, ONLY use sources explicitly verified in ./artifacts/research_data.txt file. Do NOT fabricate or assume any reference sources not documented in ./artifacts/research_data.txt.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "section_title": {
                            "type": "string",
                            "description": "The title of the section to add or update. For references sections (참고문헌), ensure all cited sources are verified in ./artifacts/research_data.txt"
                        },
                        "section_content": {
                            "type": "string",
                            "description": "The content of the section (HTML or Markdown format depending on report format). For reference sections, include only sources found in ./artifacts/research_data.txt under '### 출처:' sections"
                        },
                        "include_selected_assets": {
                            "type": "array",
                            "description": "List of specific asset filenames to include in this section (e.g., ['chart1.png', 'data.json'])",
                            "items": {"type": "string"},
                            "default": []
                        },
                        "asset_limit": {
                            "type": "integer",
                            "description": "Maximum number of assets to include in this section to avoid overwhelming content",
                            "default": 5
                        }
                    },
                    "required": ["section_title", "section_content"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "review_and_update_report",
            "description": "Review the current progressive report and update specific sections or fix issues. Allows refinement of existing content.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "section_to_update": {
                            "type": "string",
                            "description": "The section title to update, or 'overview' to get current report structure"
                        },
                        "updates": {
                            "type": "string",
                            "description": "The new content or modifications to apply to the section"
                        },
                        "action": {
                            "type": "string",
                            "description": "Type of update action",
                            "enum": ["replace", "append", "prepend", "overview"],
                            "default": "replace"
                        }
                    },
                    "required": ["section_to_update"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "finalize_progressive_report",
            "description": "Finalize the progressive report by adding any remaining assets and generating the final output files. Ensures all reference sources in the report are verified against ./artifacts/research_data.txt to prevent hallucinated citations.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "add_remaining_assets": {
                            "type": "boolean",
                            "description": "Whether to add any remaining unused assets to an appendix section",
                            "default": False
                        },
                        "generate_pdf": {
                            "type": "boolean",
                            "description": "Whether to also generate a PDF version",
                            "default": True
                        },
                        "cleanup_intermediate_files": {
                            "type": "boolean",
                            "description": "Whether to clean up intermediate working files",
                            "default": False
                        }
                    },
                    "required": []
                }
            }
        }
    }
]

def handle_generate_html_report(title: str, content: str, include_charts: bool = True) -> str:
    """Generate HTML report with embedded charts"""
    import os
    import glob
    
    try:
        # artifacts 디렉토리 생성
        os.makedirs("./artifacts", exist_ok=True)
        
        # 개선된 HTML 템플릿
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        body {{
            font-family: 'Noto Sans KR', 'Nanum Gothic', 'Malgun Gothic', sans-serif;
            margin: 2cm;
            line-height: 1.6;
            color: #333;
            background-color: #fff;
            font-size: 14px;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-weight: 700;
            font-size: 28px;
        }}
        h2 {{
            color: #3498db;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
            font-weight: 600;
            font-size: 22px;
        }}
        h3 {{
            color: #2980b9;
            margin-top: 20px;
            font-weight: 500;
            font-size: 18px;
        }}
        h4 {{
            color: #34495e;
            margin-top: 15px;
            font-weight: 500;
            font-size: 16px;
        }}
        .content {{
            margin-top: 20px;
        }}
        img {{
            max-width: 70%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-caption {{
            text-align: center;
            font-style: italic;
            margin-bottom: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 13px;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .executive-summary {{
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 5px solid #3498db;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .figure {{
            text-align: center;
            margin: 30px 0;
        }}
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        ul, ol {{
            padding-left: 20px;
            margin-bottom: 15px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #f9f9f9;
            font-style: italic;
        }}
        .header-info {{
            text-align: right;
            font-size: 12px;
            color: #666;
            margin-bottom: 20px;
        }}
        .toc {{
            margin: 20px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .toc h2 {{
            margin-top: 0;
            border-bottom: 1px solid #ddd;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            padding: 5px 0;
            border-bottom: 1px dotted #ddd;
        }}
        /* 반응형 디자인 */
        @media (max-width: 768px) {{
            body {{
                margin: 1cm;
                font-size: 12px;
            }}
            h1 {{
                font-size: 24px;
            }}
            h2 {{
                font-size: 20px;
            }}
            h3 {{
                font-size: 16px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header-info">
        생성일: {__import__('datetime').datetime.now().strftime('%Y년 %m월 %d일')}
    </div>
    
    <h1>{title}</h1>
    
    <div class="content">
        {content}
    </div>
"""
        
        # 차트 포함
        if include_charts:
            chart_files = glob.glob("./artifacts/*.png")
            if chart_files:
                html_content += "\n    <h2>생성된 차트 및 시각화</h2>\n"
                for chart_file in sorted(chart_files):
                    chart_name = os.path.basename(chart_file)
                    html_content += f"""
    <div class="figure">
        <img src="{chart_file}" alt="{chart_name}">
        <div class="image-caption">그림: {chart_name}</div>
    </div>
"""
        
        html_content += """
</body>
</html>"""
        
        # HTML 파일 저장
        html_file_path = './report.html'
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return f"HTML report successfully generated: {html_file_path}"
        
    except Exception as e:
        return f"Error generating HTML report: {str(e)}"

def handle_generate_markdown_report(title: str, content: str, include_images: bool = True) -> str:
    """Generate Markdown report with image references"""
    import os
    import glob
    
    try:
        # artifacts 디렉토리 생성
        os.makedirs("./artifacts", exist_ok=True)
        
        # 마크다운 헤더 정보
        current_date = __import__('datetime').datetime.now().strftime('%Y년 %m월 %d일')
        
        # 마크다운 컨텐츠 생성
        md_content = f"""# {title}

---

**생성일:** {current_date}

---

{content}

"""
        
        # 이미지 포함
        if include_images:
            image_files = glob.glob("./artifacts/*.png")
            if image_files:
                md_content += "## 생성된 차트 및 시각화\n\n"
                for image_file in sorted(image_files):
                    image_name = os.path.basename(image_file)
                    md_content += f"### {image_name}\n\n"
                    md_content += f"![{image_name}]({image_file})\n\n"
                    md_content += f"*그림: {image_name}*\n\n"
                    md_content += "---\n\n"
        
        # 푸터 추가
        md_content += f"""
---

*이 보고서는 {current_date}에 자동 생성되었습니다.*
"""
        
        # 마크다운 파일 저장
        md_file_path = './final_report.md'
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return f"Markdown report successfully generated: {md_file_path}"
        
    except Exception as e:
        return f"Error generating Markdown report: {str(e)}"

def handle_generate_pdf_report(title: str, content: str, include_charts: bool = True, page_size: str = "A4") -> str:
    """Generate PDF report with proper Korean font support and professional formatting"""
    import os
    import glob
    import base64
    
    try:
        # artifacts 디렉토리 생성
        os.makedirs("./artifacts", exist_ok=True)
        
        # 이미지를 Base64로 인코딩하는 함수
        def encode_image_to_base64(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
            except Exception:
                return None
        
        # 개선된 HTML 템플릿 (PDF 최적화)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        @page {{
            size: {page_size};
            margin: 2cm;
            @bottom-center {{
                content: counter(page);
                font-size: 10px;
                color: #666;
            }}
        }}
        body {{
            font-family: 'Noto Sans KR', 'Malgun Gothic', '맑은 고딕', sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #fff;
            margin: 0;
            padding: 0;
            font-size: 12px;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-weight: 700;
            font-size: 24px;
            page-break-after: avoid;
        }}
        h2 {{
            color: #3498db;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
            font-weight: 600;
            font-size: 18px;
            page-break-after: avoid;
        }}
        h3 {{
            color: #2980b9;
            margin-top: 20px;
            font-weight: 500;
            font-size: 16px;
            page-break-after: avoid;
        }}
        h4 {{
            color: #34495e;
            margin-top: 15px;
            font-weight: 500;
            font-size: 14px;
            page-break-after: avoid;
        }}
        .content {{
            margin-top: 20px;
        }}
        img {{
            max-width: 70%;
            height: auto;
            display: block;
            margin: 15px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }}
        .image-caption {{
            text-align: center;
            font-style: italic;
            margin-bottom: 20px;
            color: #666;
            font-size: 0.9em;
            margin-top: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            page-break-inside: avoid;
            font-size: 11px;
        }}
        table, th, td {{
            border: 1px solid #ddd;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .executive-summary {{
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 5px solid #3498db;
            margin: 20px 0;
            border-radius: 4px;
            page-break-inside: avoid;
        }}
        .figure {{
            text-align: center;
            margin: 30px 0;
            page-break-inside: avoid;
        }}
        p {{
            margin-bottom: 12px;
            text-align: justify;
            orphans: 3;
            widows: 3;
        }}
        ul, ol {{
            padding-left: 20px;
            margin-bottom: 15px;
        }}
        li {{
            margin-bottom: 6px;
        }}
        .page-break {{
            page-break-before: always;
        }}
        .no-break {{
            page-break-inside: avoid;
        }}
        .header-info {{
            text-align: right;
            font-size: 10px;
            color: #666;
            margin-bottom: 20px;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #f9f9f9;
            font-style: italic;
        }}
        .toc {{
            margin: 20px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .toc h2 {{
            margin-top: 0;
            border-bottom: 1px solid #ddd;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            padding: 5px 0;
            border-bottom: 1px dotted #ddd;
        }}
    </style>
</head>
<body>
    <div class="header-info">
        생성일: {__import__('datetime').datetime.now().strftime('%Y년 %m월 %d일')}
    </div>
    
    <h1>{title}</h1>
    
    <div class="content">
        {content}
    </div>
"""
        
        # 차트 포함 (Base64 인코딩으로 PDF에 임베드)
        if include_charts:
            chart_files = glob.glob("./artifacts/*.png")
            if chart_files:
                html_content += '\n    <div class="page-break"></div>\n'
                html_content += "\n    <h2>생성된 차트 및 시각화</h2>\n"
                for chart_file in sorted(chart_files):
                    chart_name = os.path.basename(chart_file)
                    base64_image = encode_image_to_base64(chart_file)
                    if base64_image:
                        html_content += f"""
    <div class="figure no-break">
        <img src="data:image/png;base64,{base64_image}" alt="{chart_name}">
        <div class="image-caption">그림: {chart_name}</div>
    </div>
"""
                    else:
                        # Base64 인코딩 실패시 파일 경로 사용
                        html_content += f"""
    <div class="figure no-break">
        <img src="{chart_file}" alt="{chart_name}">
        <div class="image-caption">그림: {chart_name}</div>
    </div>
"""
        
        html_content += """
</body>
</html>"""
        
        # HTML 파일 저장
        html_file_path = './report_for_pdf.html'
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # PDF 생성 시도 (weasyprint 사용)
        pdf_file_path = './final_report.pdf'
        try:
            import weasyprint
            # CSS와 이미지를 제대로 로드하기 위해 base_url 설정
            weasyprint.HTML(string=html_content, base_url='.').write_pdf(pdf_file_path)
            return f"PDF report successfully generated: {pdf_file_path}\\nIntermediate HTML file: {html_file_path}"
        except ImportError:
            # weasyprint가 없으면 wkhtmltopdf 사용 시도
            try:
                import pdfkit
                options = {
                    'page-size': page_size,
                    'margin-top': '2cm',
                    'margin-right': '2cm',
                    'margin-bottom': '2cm',
                    'margin-left': '2cm',
                    'encoding': "UTF-8",
                    'no-outline': None,
                    'enable-local-file-access': None
                }
                pdfkit.from_string(html_content, pdf_file_path, options=options)
                return f"PDF report successfully generated: {pdf_file_path}\\nIntermediate HTML file: {html_file_path}"
            except ImportError:
                # PDF 라이브러리가 없으면 HTML만 생성
                return f"PDF libraries not available. HTML report generated: {html_file_path}\\nTo generate PDF, install weasyprint: pip install weasyprint"
        except Exception as e:
            return f"Error generating PDF (HTML available): {html_file_path}\\nPDF Error: {str(e)}\\nTo fix, install weasyprint: pip install weasyprint"
        
    except Exception as e:
        return f"Error generating PDF report: {str(e)}"

def handle_estimate_bedrock_costs(query_complexity: str = "medium", execution_count: int = 1, include_analysis: bool = True) -> str:
    """Estimate AWS Bedrock costs for agentic program execution"""
    import os
    from datetime import datetime
    
    try:
        # Define cost per 1K tokens for different models (as of 2024)
        model_costs = {
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015}
        }
        
        # Estimate token usage based on query complexity
        complexity_multipliers = {
            "simple": {"researcher": 5000, "coder": 3000, "reporter": 4000},
            "medium": {"researcher": 15000, "coder": 10000, "reporter": 12000},
            "complex": {"researcher": 30000, "coder": 25000, "reporter": 20000},
            "very_complex": {"researcher": 50000, "coder": 40000, "reporter": 35000}
        }
        
        # Agent model mapping
        agent_models = {
            "researcher": "claude-3-5-sonnet",
            "coder": "claude-3-5-sonnet", 
            "reporter": "claude-3-5-sonnet"
        }
        
        tokens = complexity_multipliers.get(query_complexity, complexity_multipliers["medium"])
        
        total_cost = 0
        cost_breakdown = {}
        
        for agent, token_count in tokens.items():
            model = agent_models[agent]
            input_tokens = token_count * 0.7  # Assume 70% input tokens
            output_tokens = token_count * 0.3  # Assume 30% output tokens
            
            input_cost = (input_tokens / 1000) * model_costs[model]["input"]
            output_cost = (output_tokens / 1000) * model_costs[model]["output"]
            agent_cost = (input_cost + output_cost) * execution_count
            
            cost_breakdown[agent] = {
                "model": model,
                "input_tokens": int(input_tokens * execution_count),
                "output_tokens": int(output_tokens * execution_count),
                "input_cost": input_cost * execution_count,
                "output_cost": output_cost * execution_count,
                "total_cost": agent_cost
            }
            total_cost += agent_cost
        
        # Generate cost report
        cost_report = f"""AWS Bedrock Cost Estimation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Query Complexity: {query_complexity.upper()}
Execution Count: {execution_count}

COST BREAKDOWN BY AGENT:
"""
        
        for agent, costs in cost_breakdown.items():
            cost_report += f"""
{agent.upper()} Agent:
  Model: {costs['model']}
  Input Tokens: {costs['input_tokens']:,}
  Output Tokens: {costs['output_tokens']:,}
  Input Cost: ${costs['input_cost']:.4f}
  Output Cost: ${costs['output_cost']:.4f}
  Agent Total: ${costs['total_cost']:.4f}
"""
        
        cost_report += f"""
TOTAL ESTIMATED COST: ${total_cost:.4f}

COST PROJECTIONS:
Daily (10 executions): ${total_cost * 10:.2f}
Weekly (50 executions): ${total_cost * 50:.2f}
Monthly (200 executions): ${total_cost * 200:.2f}

NOTE: These are estimates based on average usage patterns.
Actual costs may vary based on actual token consumption,
model performance, and AWS pricing changes.
"""
        
        if include_analysis:
            cost_report += f"""

COST OPTIMIZATION RECOMMENDATIONS:
1. Use Claude-3-Haiku for simple tasks to reduce costs by ~80%
2. Implement token usage monitoring and optimization
3. Cache frequently used results to avoid redundant API calls
4. Consider batch processing for multiple queries
5. Monitor actual vs estimated costs and adjust complexity ratings

RISK FACTORS:
- Complex queries may exceed token estimates by 20-50%
- Multi-turn conversations can increase costs significantly
- Tool usage and reasoning steps add to token consumption
- Peak usage periods may affect response times but not costs
"""
        
        # Save to artifacts
        os.makedirs("./artifacts", exist_ok=True)
        with open("./artifacts/bedrock_billing.txt", "w", encoding="utf-8") as f:
            f.write(cost_report)
        
        return f"Cost estimation completed. Total estimated cost: ${total_cost:.4f} per execution. Detailed report saved to ./artifacts/bedrock_billing.txt"
        
    except Exception as e:
        return f"Error generating cost estimate: {str(e)}"

def handle_initialize_progressive_report(title: str, format: str = "html", sections: List[str] = None) -> str:
    """Initialize a progressive report with basic structure"""
    import os
    from datetime import datetime
    
    try:
        if sections is None:
            sections = ["Executive Summary", "Analysis Results", "Key Findings", "Recommendations", "Conclusion"]
        
        # Create artifacts directory
        os.makedirs("./artifacts", exist_ok=True)
        
        current_date = datetime.now().strftime('%Y년 %m월 %d일')
        
        if format == "html":
            # Create HTML template
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        body {{
            font-family: 'Noto Sans KR', 'Nanum Gothic', 'Malgun Gothic', sans-serif;
            margin: 2cm;
            line-height: 1.6;
            color: #333;
            background-color: #fff;
            font-size: 14px;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-weight: 700;
            font-size: 28px;
        }}
        h2 {{
            color: #3498db;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
            font-weight: 600;
            font-size: 22px;
        }}
        h3 {{
            color: #2980b9;
            margin-top: 20px;
            font-weight: 500;
            font-size: 18px;
        }}
        .content {{
            margin: 20px 0;
        }}
        img {{
            max-width: 70%;
            height: auto;
            display: block;
            margin: 15px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-caption {{
            text-align: center;
            font-style: italic;
            margin-bottom: 20px;
            color: #666;
            font-size: 0.9em;
            margin-top: 8px;
        }}
        .section-placeholder {{
            background-color: #f8f9fa;
            border: 2px dashed #3498db;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header-info" style="text-align: right; font-size: 12px; color: #666; margin-bottom: 20px;">
        생성일: {current_date}
    </div>
    
    <h1>{title}</h1>
    
    <!-- REPORT_CONTENT_START -->
"""
            
            # Add section placeholders
            for section in sections:
                html_content += f"""
    <h2>{section}</h2>
    <div class="section-placeholder">
        <p>📝 {section} 섹션이 작성될 예정입니다.</p>
        <p><em>이 섹션은 add_section_to_report 도구를 사용하여 내용을 추가할 수 있습니다.</em></p>
    </div>
"""
            
            html_content += """
    <!-- REPORT_CONTENT_END -->
    
    <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ccc; color: #666; font-size: 0.9em;">
        <em>본 보고서는 점진적으로 작성되고 있습니다.</em>
    </div>
</body>
</html>"""
            
            file_path = "./artifacts/progressive_report.html"
            
        else:  # markdown format
            md_content = f"""# {title}

---

**생성일:** {current_date}

---

"""
            # Add section placeholders
            for section in sections:
                md_content += f"""
## {section}

> 📝 {section} 섹션이 작성될 예정입니다.
> 
> *이 섹션은 add_section_to_report 도구를 사용하여 내용을 추가할 수 있습니다.*

---

"""
            
            md_content += f"""

---

*본 보고서는 점진적으로 작성되고 있습니다.*
"""
            
            file_path = "./artifacts/progressive_report.md"
        
        # Save the initial report
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content if format == "html" else md_content)
        
        # Save metadata
        metadata = {
            "title": title,
            "format": format,
            "sections": sections,
            "created_date": current_date,
            "file_path": file_path,
            "completed_sections": [],
            "used_assets": []
        }
        
        import json
        with open("./artifacts/progressive_report_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return f"Progressive report initialized: {file_path}. Sections: {', '.join(sections)}"
        
    except Exception as e:
        return f"Error initializing progressive report: {str(e)}"

def handle_add_section_to_report(section_title: str, section_content: str, include_selected_assets: List[str] = None, asset_limit: int = 5) -> str:
    """Add or update a specific section in the progressive report"""
    import os
    import json
    import glob
    import re
    
    try:
        if include_selected_assets is None:
            include_selected_assets = []
        
        # Load metadata
        metadata_path = "./artifacts/progressive_report_metadata.json"
        if not os.path.exists(metadata_path):
            return "Error: Progressive report not initialized. Please use initialize_progressive_report first."
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        report_path = metadata["file_path"]
        format = metadata["format"]
        
        # Read current report
        with open(report_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        # Find and include relevant assets
        all_assets = glob.glob("./artifacts/*")
        relevant_assets = []
        
        if include_selected_assets:
            # Use specifically requested assets
            for asset_name in include_selected_assets:
                asset_path = f"./artifacts/{asset_name}"
                if os.path.exists(asset_path):
                    relevant_assets.append(asset_path)
        else:
            # Auto-select relevant assets based on section title keywords
            keywords = section_title.lower().split()
            for asset_path in all_assets:
                if asset_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.json')):
                    asset_name = os.path.basename(asset_path).lower()
                    if any(keyword in asset_name for keyword in keywords):
                        relevant_assets.append(asset_path)
        
        # Limit assets to avoid overwhelming content
        relevant_assets = relevant_assets[:asset_limit]
        
        # Build section content with assets
        if format == "html":
            # 중복 제목 방지: section_content에 이미 h2 태그가 있는지 확인
            content_has_h2 = re.search(rf'<h2.*?>{re.escape(section_title)}.*?</h2>', section_content, re.IGNORECASE)
            
            if content_has_h2:
                # 이미 제목이 있으면 제목 없이 content만 사용
                full_section_content = f"""
    <div class="content">
        {section_content}
    </div>
"""
            else:
                # 제목이 없으면 제목 추가
                full_section_content = f"""
    <h2>{section_title}</h2>
    <div class="content">
        {section_content}
    </div>
"""
            
            # Add images with appropriate sizing
            for asset_path in relevant_assets:
                if asset_path.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    asset_name = os.path.basename(asset_path)
                    full_section_content += f"""
    <div style="text-align: center; margin: 15px 0;">
        <img src="{asset_path}" alt="{asset_name}" style="max-width: 70%; height: auto; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div class="image-caption" style="font-size: 0.9em; color: #666; margin-top: 8px;">그림: {asset_name}</div>
    </div>
"""
        else:  # markdown
            # 중복 제목 방지: section_content에 이미 ## 제목이 있는지 확인
            content_has_h2 = re.search(rf'^## {re.escape(section_title)}\s*$', section_content, re.MULTILINE)
            
            if content_has_h2:
                # 이미 제목이 있으면 content만 사용
                full_section_content = f"""
{section_content}

"""
            else:
                # 제목이 없으면 제목 추가
                full_section_content = f"""
## {section_title}

{section_content}

"""
            # Add images
            for asset_path in relevant_assets:
                if asset_path.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    asset_name = os.path.basename(asset_path)
                    full_section_content += f"""
<div style="text-align: center; margin: 15px 0;">
<img src="{asset_path}" alt="{asset_name}" style="max-width: 70%; height: auto;">
</div>

*그림: {asset_name}*

"""
        
        # Update the report content
        if format == "html":
            # Find existing section or placeholder
            section_pattern = rf'<h2>{re.escape(section_title)}</h2>.*?(?=<h2>|<!-- REPORT_CONTENT_END -->)'
            placeholder_pattern = rf'<h2>{re.escape(section_title)}</h2>\s*<div class="section-placeholder">.*?</div>'
            
            if re.search(section_pattern, current_content, re.DOTALL):
                # Replace existing section
                current_content = re.sub(section_pattern, full_section_content.strip(), current_content, flags=re.DOTALL)
            elif re.search(placeholder_pattern, current_content, re.DOTALL):
                # Replace placeholder
                current_content = re.sub(placeholder_pattern, full_section_content.strip(), current_content, flags=re.DOTALL)
            else:
                # Add new section before end marker
                current_content = current_content.replace(
                    "<!-- REPORT_CONTENT_END -->", 
                    full_section_content + "\n    <!-- REPORT_CONTENT_END -->"
                )
        else:  # markdown
            # Find existing section
            section_pattern = rf'^## {re.escape(section_title)}.*?(?=^## |\Z)'
            
            if re.search(section_pattern, current_content, re.MULTILINE | re.DOTALL):
                # Replace existing section
                current_content = re.sub(section_pattern, full_section_content.strip() + "\n\n", current_content, flags=re.MULTILINE | re.DOTALL)
            else:
                # Add new section before final note
                final_note_pattern = r'\n---\n\*본 보고서는 점진적으로 작성되고 있습니다\.\*'
                if re.search(final_note_pattern, current_content):
                    current_content = re.sub(final_note_pattern, f"\n{full_section_content}\n---\n\n*본 보고서는 점진적으로 작성되고 있습니다.*", current_content)
                else:
                    current_content += f"\n{full_section_content}"
        
        # Save updated report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(current_content)
        
        # Update metadata
        if section_title not in metadata["completed_sections"]:
            metadata["completed_sections"].append(section_title)
        
        for asset in relevant_assets:
            asset_name = os.path.basename(asset)
            if asset_name not in metadata["used_assets"]:
                metadata["used_assets"].append(asset_name)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return f"Section '{section_title}' added to report. Included {len(relevant_assets)} assets. Report updated: {report_path}"
        
    except Exception as e:
        return f"Error adding section to report: {str(e)}"

def handle_review_and_update_report(section_to_update: str, updates: str = "", action: str = "replace") -> str:
    """Review the current progressive report and update sections"""
    import os
    import json
    import re
    
    try:
        # Load metadata
        metadata_path = "./artifacts/progressive_report_metadata.json"
        if not os.path.exists(metadata_path):
            return "Error: Progressive report not initialized."
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        report_path = metadata["file_path"]
        
        # Read current report
        with open(report_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        if action == "overview" or section_to_update.lower() == "overview":
            # Provide overview of current report structure
            completed_sections = metadata.get("completed_sections", [])
            used_assets = metadata.get("used_assets", [])
            total_sections = len(metadata.get("sections", []))
            
            # 중복 제목 검사 추가
            format = metadata["format"]
            duplicate_titles = []
            
            if format == "html":
                # HTML에서 h2 태그들 찾기
                h2_titles = re.findall(r'<h2[^>]*>(.*?)</h2>', current_content, re.IGNORECASE)
                title_counts = {}
                for title in h2_titles:
                    title_clean = re.sub(r'<[^>]+>', '', title).strip()  # HTML 태그 제거
                    title_counts[title_clean] = title_counts.get(title_clean, 0) + 1
                duplicate_titles = [title for title, count in title_counts.items() if count > 1]
            else:
                # Markdown에서 ## 제목들 찾기
                h2_titles = re.findall(r'^## (.+)$', current_content, re.MULTILINE)
                title_counts = {}
                for title in h2_titles:
                    title_counts[title.strip()] = title_counts.get(title.strip(), 0) + 1
                duplicate_titles = [title for title, count in title_counts.items() if count > 1]
            
            overview = f"""📊 Progressive Report Overview:

Title: {metadata['title']}
Format: {metadata['format']}
File: {report_path}
Created: {metadata['created_date']}

Progress: {len(completed_sections)}/{total_sections} sections completed
Completed Sections: {', '.join(completed_sections) if completed_sections else 'None'}
Remaining Sections: {', '.join([s for s in metadata['sections'] if s not in completed_sections])}

Assets Used: {len(used_assets)} files
Used Assets: {', '.join(used_assets[:10])}{'...' if len(used_assets) > 10 else ''}

Report Size: {len(current_content)} characters

⚠️ Duplicate Titles Check:
{f'Found duplicate titles: {", ".join(duplicate_titles)}' if duplicate_titles else '✅ No duplicate titles found'}
"""
            return overview
        
        # Update specific section
        format = metadata["format"]
        
        if format == "html":
            section_pattern = rf'(<h2>{re.escape(section_to_update)}</h2>.*?)(?=<h2>|<!-- REPORT_CONTENT_END -->)'
        else:
            section_pattern = rf'(^## {re.escape(section_to_update)}.*?)(?=^## |\Z)'
        
        match = re.search(section_pattern, current_content, re.DOTALL | re.MULTILINE)
        
        if not match:
            return f"Section '{section_to_update}' not found in report."
        
        original_section = match.group(1)
        
        if action == "replace":
            if format == "html":
                new_section = f"<h2>{section_to_update}</h2>\n    <div class=\"content\">\n        {updates}\n    </div>"
            else:
                new_section = f"## {section_to_update}\n\n{updates}\n"
        elif action == "append":
            if format == "html":
                new_section = original_section.rstrip() + f"\n        {updates}\n    </div>" if original_section.endswith("</div>") else original_section + f"\n        {updates}"
            else:
                new_section = original_section.rstrip() + f"\n\n{updates}"
        elif action == "prepend":
            if format == "html":
                new_section = original_section.replace(
                    f"<h2>{section_to_update}</h2>\n    <div class=\"content\">",
                    f"<h2>{section_to_update}</h2>\n    <div class=\"content\">\n        {updates}"
                )
            else:
                lines = original_section.split('\n')
                lines.insert(2, updates)  # Insert after title and empty line
                new_section = '\n'.join(lines)
        
        # Apply the update
        current_content = current_content.replace(original_section, new_section)
        
        # Save updated report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(current_content)
        
        return f"Section '{section_to_update}' updated with action '{action}'. Report saved."
        
    except Exception as e:
        return f"Error reviewing/updating report: {str(e)}"

def handle_finalize_progressive_report(add_remaining_assets: bool = False, generate_pdf: bool = True, cleanup_intermediate_files: bool = False) -> str:
    """Finalize the progressive report"""
    import os
    import json
    import glob
    import subprocess
    import re
    from datetime import datetime
    
    try:
        # Load metadata
        metadata_path = "./artifacts/progressive_report_metadata.json"
        if not os.path.exists(metadata_path):
            return "Error: Progressive report not initialized."
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        report_path = metadata["file_path"]
        format = metadata["format"]
        used_assets = set(metadata.get("used_assets", []))
        
        # Add remaining assets if requested
        if add_remaining_assets:
            all_image_assets = glob.glob("./artifacts/*.png") + glob.glob("./artifacts/*.jpg") + glob.glob("./artifacts/*.jpeg") + glob.glob("./artifacts/*.gif")
            remaining_assets = [asset for asset in all_image_assets if os.path.basename(asset) not in used_assets]
            
            if remaining_assets:
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if format == "html":
                    appendix_content = """
    <h2>부록: 추가 생성된 자료</h2>
    <div class="content">
        <p>분석 과정에서 생성된 추가 차트 및 데이터 파일들입니다.</p>
"""
                    for asset in remaining_assets[:10]:  # Limit to 10 additional assets
                        asset_name = os.path.basename(asset)
                        appendix_content += f"""
        <div style="text-align: center; margin: 20px 0;">
            <img src="{asset}" alt="{asset_name}">
            <div class="image-caption">그림: {asset_name}</div>
        </div>
"""
                    appendix_content += "\n    </div>"
                    
                    content = content.replace("<!-- REPORT_CONTENT_END -->", appendix_content + "\n    <!-- REPORT_CONTENT_END -->")
                
                else:  # markdown
                    appendix_content = "\n## 부록: 추가 생성된 자료\n\n분석 과정에서 생성된 추가 차트 및 데이터 파일들입니다.\n\n"
                    for asset in remaining_assets[:10]:
                        asset_name = os.path.basename(asset)
                        appendix_content += f"\n![{asset_name}]({asset})\n\n*그림: {asset_name}*\n\n"
                    
                    final_note_pattern = r'\n---\n\*본 보고서는 점진적으로 작성되고 있습니다\.\*'
                    content = re.sub(final_note_pattern, f"{appendix_content}\n---\n\n*본 보고서가 완성되었습니다.*", content)
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        
        # Generate final files
        final_files = [report_path]
        
        # Copy to standard names
        if format == "html":
            final_html_path = "./report.html"
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove "progressive" indicators
            content = content.replace("본 보고서는 점진적으로 작성되고 있습니다.", "본 보고서가 완성되었습니다.")
            
            with open(final_html_path, 'w', encoding='utf-8') as f:
                f.write(content)
            final_files.append(final_html_path)
        else:
            final_md_path = "./final_report.md"
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = content.replace("본 보고서는 점진적으로 작성되고 있습니다.", "본 보고서가 완성되었습니다.")
            
            with open(final_md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            final_files.append(final_md_path)
        
        # Generate PDF if requested
        if generate_pdf:
            try:
                if format == "markdown":
                    source_file = final_files[-1]
                else:
                    # Convert HTML to markdown first for PDF generation
                    source_file = "./temp_for_pdf.md"
                    # Simple HTML to markdown conversion
                    with open(report_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Basic HTML to markdown conversion
                    md_content = html_content
                    md_content = re.sub(r'<h1>(.*?)</h1>', r'# \1', md_content)
                    md_content = re.sub(r'<h2>(.*?)</h2>', r'## \1', md_content)
                    md_content = re.sub(r'<h3>(.*?)</h3>', r'### \1', md_content)
                    md_content = re.sub(r'<p>(.*?)</p>', r'\1\n', md_content, flags=re.DOTALL)
                    md_content = re.sub(r'<div.*?>', '', md_content)
                    md_content = re.sub(r'</div>', '', md_content)
                    md_content = re.sub(r'<style>.*?</style>', '', md_content, flags=re.DOTALL)
                    md_content = re.sub(r'<.*?>', '', md_content)
                    
                    with open(source_file, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                
                pdf_file_path = './final_report.pdf'
                pandoc_cmd = f'pandoc "{source_file}" -o "{pdf_file_path}" --pdf-engine=xelatex -V mainfont="NanumGothic" -V geometry="margin=0.5in"'
                
                result = subprocess.run(pandoc_cmd, shell=True, check=True, capture_output=True)
                final_files.append(pdf_file_path)
                
                if source_file == "./temp_for_pdf.md":
                    os.remove(source_file)
                    
            except Exception as e:
                print(f"PDF generation failed: {str(e)}")
        
        # Cleanup if requested
        if cleanup_intermediate_files:
            try:
                os.remove(metadata_path)
                if os.path.exists("./artifacts/progressive_report.html"):
                    os.remove("./artifacts/progressive_report.html")
                if os.path.exists("./artifacts/progressive_report.md"):
                    os.remove("./artifacts/progressive_report.md")
            except:
                pass
        
        # Update metadata
        metadata["finalized"] = True
        metadata["finalized_date"] = datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')
        metadata["final_files"] = final_files
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 명확한 완성 신호 포함
        completion_message = f"🎉 REPORT_GENERATION_COMPLETED_SUCCESSFULLY 🎉\n\nProgressive report finalized! Generated files: {', '.join(final_files)}\n\nFinal report generation is now complete. The workflow should terminate."
        
        return completion_message
        
    except Exception as e:
        return f"Error finalizing report: {str(e)}"

reporter_tool_config = {
    "tools": tool_list,
    # "toolChoice": {
    #    "tool": {
    #        "name": "summarize_email"
    #    }
    # }
}

def process_reporter_tool(tool) -> str:
    """Process a tool invocation
    
    Args:
        tool_name: Name of the tool to invoke
        tool_input: Input parameters for the tool
        
    Returns:
        Result of the tool invocation as a string
    """
    
    tool_name, tool_input = tool["name"], tool["input"]
    
    if tool_name == "python_repl_tool":
        # Create a new instance of the Tavily search tool
        results = handle_python_repl_tool(code=tool_input["code"])
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
        #return response
    elif tool_name == "bash_tool":
        results = handle_bash_tool(cmd=tool_input["cmd"])
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "generate_html_report":
        results = handle_generate_html_report(
            title=tool_input["title"],
            content=tool_input["content"],
            include_charts=tool_input.get("include_charts", True)
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "generate_markdown_report":
        results = handle_generate_markdown_report(
            title=tool_input["title"],
            content=tool_input["content"],
            include_images=tool_input.get("include_images", True)
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "generate_pdf_report":
        results = handle_generate_pdf_report(
            title=tool_input["title"],
            content=tool_input["content"],
            include_charts=tool_input.get("include_charts", True),
            page_size=tool_input.get("page_size", "A4")
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "estimate_bedrock_costs":
        results = handle_estimate_bedrock_costs(
            query_complexity=tool_input["query_complexity"],
            execution_count=tool_input["execution_count"],
            include_analysis=tool_input["include_analysis"]
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "initialize_progressive_report":
        results = handle_initialize_progressive_report(
            title=tool_input["title"],
            format=tool_input.get("format", "html"),
            sections=tool_input.get("sections", None)
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "add_section_to_report":
        results = handle_add_section_to_report(
            section_title=tool_input["section_title"],
            section_content=tool_input["section_content"],
            include_selected_assets=tool_input.get("include_selected_assets", []),
            asset_limit=tool_input.get("asset_limit", 5)
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "review_and_update_report":
        results = handle_review_and_update_report(
            section_to_update=tool_input["section_to_update"],
            updates=tool_input.get("updates", ""),
            action=tool_input.get("action", "replace")
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    elif tool_name == "finalize_progressive_report":
        results = handle_finalize_progressive_report(
            add_remaining_assets=tool_input.get("add_remaining_assets", False),
            generate_pdf=tool_input.get("generate_pdf", True),
            cleanup_intermediate_files=tool_input.get("cleanup_intermediate_files", False)
        )
        tool_result = {
            "toolUseId": tool['toolUseId'],
            "content": [{"json": {"text": results}}]
        }
    else:
        print (f"Unknown tool: {tool_name}")
        
    results = {"role": "user","content": [{"toolResult": tool_result}]}
    
    return results