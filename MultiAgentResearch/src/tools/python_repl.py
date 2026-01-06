import time
import logging
from typing import Annotated
from langchain_core.tools import tool
#from langchain_experimental.utilities import PythonREPL
from .decorators import log_io
import os
import sys
import io
import contextlib
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용하지 않도록 설정
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize REPL and logger
#repl = PythonREPL()

# 새 핸들러와 포맷터 설정
logger = logging.getLogger(__name__)
logger.propagate = False  # 상위 로거로 메시지 전파 중지
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter('\n%(levelname)s [%(name)s] %(message)s')  # 로그 레벨이 동적으로 표시되도록 변경
handler.setFormatter(formatter)
logger.addHandler(handler)
# DEBUG와 INFO 중 원하는 레벨로 설정
logger.setLevel(logging.INFO)  # 기본 레벨은 INFO로 설정

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


import subprocess

class PythonREPL:
    def __init__(self):
        # 기본 모듈들을 전역 네임스페이스에 추가
        self.globals = {
            'os': os,
            'sys': sys,
            'io': io,
            'plt': plt,
            'datetime': datetime,
            'subprocess': subprocess,
            'matplotlib': matplotlib,
            'time': time,
            'logging': logging
        }
        
    def run(self, command):
        try:
            # 출력을 캡처하기 위한 StringIO 객체 생성
            output = io.StringIO()
            
            # 표준 출력을 리다이렉트
            with contextlib.redirect_stdout(output):
                # 코드 실행을 위한 준비
                code = f"""
import os
import sys
import io
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time
import logging
import subprocess

{command}
"""
                # 코드 실행
                exec(code, self.globals)
            
            # 결과 반환
            return output.getvalue()
        except Exception as e:
            return f"Error: {str(e)}"

repl = PythonREPL()

@log_io
def handle_python_repl_tool(code: str) -> str:
    """Execute Python code and return the output."""
    logger.info(f"{Colors.GREEN}===== Executing Python code ====={Colors.END}")
    
    # Create artifacts directory if it doesn't exist
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    try:
        # Execute the code using PythonREPL
        result = repl.run(code)
        
        # Determine if code is likely for plotting once
        save_code_if_plotting = any(keyword in code for keyword in ["plt.", "plot", "chart", "graph"])
        
        # If there are any active plots, save them and the code
        if plt.get_fignums():
            # The previous logic for saving code once as chart_code_{timestamp}.py is removed.
            
            # Save the plots and their corresponding code
            for i, fig_num in enumerate(plt.get_fignums()):
                fig = plt.figure(fig_num)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Define base filename (without extension) for both image and code file
                base_filename = f"chart_{timestamp}_{i+1}"
                
                # Save the plot (image)
                fig_path = os.path.join(artifacts_dir, f"{base_filename}.png")
                fig.savefig(fig_path)
                plt.close(fig) # Close the figure to free memory
                logger.info(f"{Colors.GREEN}Chart saved to: {fig_path}{Colors.END}")
                result += f"\nChart saved to: {fig_path}"

                # Save the python code with the same base name, if plotting keywords were found
                if save_code_if_plotting:
                    code_file_path = os.path.join(artifacts_dir, f"{base_filename}.py")
                    with open(code_file_path, "w", encoding="utf-8") as f:
                        f.write(code) # Save the original, complete input code
                    logger.info(f"{Colors.GREEN}Chart code saved to: {code_file_path}{Colors.END}")
        
        logger.info(f"{Colors.GREEN}===== Code execution successful ====={Colors.END}")
        return f"Successfully executed:\n||```python\n{code}\n```\n||Stdout: {result}"
        
    except Exception as e:
        error_msg = f"Failed to execute. Error: {repr(e)}"
        logger.debug(f"{Colors.RED}{error_msg}{Colors.END}")
        return error_msg

