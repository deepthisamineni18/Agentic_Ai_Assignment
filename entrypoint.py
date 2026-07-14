import sys
import subprocess

if __name__ == "__main__":
    args = sys.argv[1:]
    # If "--pipeline" is specified, route to cli.py, otherwise route to main.py
    if "--pipeline" in args:
        module = "research_pipeline.cli"
    else:
        module = "research_pipeline.main"
        
    cmd = [sys.executable, "-m", module] + args
    res = subprocess.run(cmd)
    sys.exit(res.returncode)
