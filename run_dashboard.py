import subprocess
import sys


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "dashboard/streamlit_app.py",
            "--server.address=0.0.0.0",
            "--server.port=8501",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
