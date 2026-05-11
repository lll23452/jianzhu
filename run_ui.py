"""启动 Streamlit 前端"""
import subprocess
import sys
import os

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', os.path.join(root, 'ui', 'app.py')])
