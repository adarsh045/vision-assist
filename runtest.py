import sys
import subprocess

NAMESPACES = {
    "normal": {
        "path": "tests/normal/",
        "pytest_args": ["-vv"]
    }
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python runtests.py <namespace1> [<namespace2> ...]")
        print("Available namespaces:", ", ".join(NAMESPACES.keys()))
        sys.exit(1)

    selected_namespaces = sys.argv[1:]

    for ns in selected_namespaces:
        if ns not in NAMESPACES:
            print(f"Unknown namespace: {ns}")
            continue

        config = NAMESPACES[ns]
        path = config["path"]
        args = config["pytest_args"]

        print(f"\n=== Running tests for namespace '{ns}' ({path}) ===\n")
        
        subprocess.run(["pytest", path] + args)


if __name__ == "__main__":
    main()
