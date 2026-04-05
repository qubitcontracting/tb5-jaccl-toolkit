"""Benchmark task definitions for the agent harness."""

TASKS = {
    "static_site_generator": {
        "name": "Static Site Generator",
        "description": "Build a CLI static site generator",
        "prompt": """Build a CLI static site generator in Python. Requirements:

1. Process Markdown files with YAML frontmatter (title, date, tags)
2. Convert Markdown to HTML using the `markdown` library
3. Generate an index page sorted by date
4. Support a --watch flag using watchdog for file monitoring
5. Clean project structure: separate modules for parsing, rendering, and CLI
6. Include unit tests for the parsing module
7. Only dependencies: standard library + markdown + pyyaml + watchdog

Write all files using write_file, then run the tests to verify.""",
        "expected_files": [
            "parser.py", "renderer.py", "cli.py", "test_parser.py", "requirements.txt"
        ],
        "checks": {
            "has_frontmatter_parsing": lambda files: any("yaml" in c.lower() and "safe_load" in c.lower() for c in files.values()),
            "has_markdown_conversion": lambda files: any("markdown" in c.lower() and "html" in c.lower() for c in files.values()),
            "has_index_generation": lambda files: any("index" in c.lower() and "sort" in c.lower() for c in files.values()),
            "has_watch_flag": lambda files: any("watch" in c.lower() and "observer" in c.lower() for c in files.values()),
            "has_tests": lambda files: any("test_" in name or "unittest" in c.lower() or "pytest" in c.lower() for name, c in files.items()),
            "has_requirements": lambda files: any("requirements" in name for name in files),
            "has_argparse": lambda files: any("argparse" in c.lower() for c in files.values()),
        },
    },

    "rest_api": {
        "name": "REST API with SQLite",
        "description": "Build a REST API for a todo list with SQLite",
        "prompt": """Build a REST API for a todo list in Python using only the standard library (http.server, json, sqlite3). Requirements:

1. SQLite database for persistence
2. Endpoints: GET /todos, POST /todos, PUT /todos/{id}, DELETE /todos/{id}
3. JSON request/response format
4. Proper HTTP status codes (200, 201, 404, 400)
5. Include unit tests that verify all CRUD operations
6. Database initialization on first run

Write all files using write_file, install any deps with shell, then run the tests.""",
        "expected_files": ["server.py", "database.py", "test_api.py"],
        "checks": {
            "has_sqlite": lambda files: any("sqlite3" in c for c in files.values()),
            "has_http_server": lambda files: any("http.server" in c or "HTTPServer" in c for c in files.values()),
            "has_get": lambda files: any("GET" in c for c in files.values()),
            "has_post": lambda files: any("POST" in c for c in files.values()),
            "has_put": lambda files: any("PUT" in c for c in files.values()),
            "has_delete": lambda files: any("DELETE" in c for c in files.values()),
            "has_tests": lambda files: any("test_" in name for name in files),
            "has_json": lambda files: any("json" in c.lower() for c in files.values()),
        },
    },

    "data_pipeline": {
        "name": "Data Pipeline",
        "description": "CSV ingestion, transformation, JSON output",
        "prompt": """Build a data processing pipeline in Python. Requirements:

1. Read CSV files from an input directory
2. Transform data: clean missing values, normalize numeric columns, add computed fields
3. Output results as JSON files
4. Configurable via a YAML config file (input_dir, output_dir, transformations)
5. Error handling for malformed CSV data (log errors, skip bad rows)
6. Include unit tests with sample CSV data
7. Only dependencies: standard library + pyyaml

Write all files using write_file, then run the tests to verify.""",
        "expected_files": ["pipeline.py", "config.yaml", "test_pipeline.py"],
        "checks": {
            "has_csv_reading": lambda files: any("csv" in c.lower() for c in files.values()),
            "has_json_output": lambda files: any("json.dump" in c or "json.dumps" in c for c in files.values()),
            "has_yaml_config": lambda files: any("config" in name and "yaml" in name for name in files),
            "has_error_handling": lambda files: any("try" in c and "except" in c for c in files.values()),
            "has_normalization": lambda files: any("normaliz" in c.lower() or "clean" in c.lower() for c in files.values()),
            "has_tests": lambda files: any("test_" in name for name in files),
        },
    },

    "cli_tool": {
        "name": "CLI Tool with Subcommands",
        "description": "Build a git-like CLI tool",
        "prompt": """Build a note-taking CLI tool in Python with subcommands. Requirements:

1. Subcommands: init, add, list, search, delete
2. `init` creates a .notes directory in current folder
3. `add "note title" "note content"` saves a note as JSON with timestamp
4. `list` shows all notes sorted by date
5. `search <query>` searches note titles and content
6. `delete <id>` removes a note
7. File-based storage (one JSON file per note in .notes/)
8. Use argparse with subparsers
9. Include unit tests for all subcommands
10. Only standard library

Write all files using write_file, then run the tests to verify.""",
        "expected_files": ["notes.py", "test_notes.py"],
        "checks": {
            "has_argparse_subparsers": lambda files: any("add_subparsers" in c or "subparsers" in c for c in files.values()),
            "has_init": lambda files: any("init" in c.lower() and "mkdir" in c.lower() or "makedirs" in c for c in files.values()),
            "has_add": lambda files: any("add" in c.lower() and ("def " in c or "add_parser" in c.lower() or "argparse" in c.lower()) for c in files.values()),
            "has_list": lambda files: any("def list" in c.lower() or "'list'" in c for c in files.values()),
            "has_search": lambda files: any("search" in c.lower() for c in files.values()),
            "has_delete": lambda files: any("delete" in c.lower() or "remove" in c.lower() for c in files.values()),
            "has_json_storage": lambda files: any("json.dump" in c for c in files.values()),
            "has_tests": lambda files: any("test_" in name for name in files),
        },
    },

    "algorithm_no_tools": {
        "name": "A* Pathfinding (No Tools)",
        "description": "Implement A* pathfinding - measures raw reasoning",
        "prompt": """Implement the A* pathfinding algorithm in Python. Requirements:

1. Grid-based pathfinding with obstacles
2. Support diagonal movement (8-directional)
3. Use Manhattan distance as heuristic for 4-dir, Chebyshev for 8-dir
4. Return the shortest path as a list of (row, col) tuples
5. Handle edge cases: no path exists, start==end, start/end on obstacle
6. Include comprehensive unit tests with multiple grid configurations
7. Only standard library (use heapq for priority queue)

Write all files using write_file, then run the tests to verify.

Do NOT use search_docs for this task - implement from your own knowledge.""",
        "expected_files": ["astar.py", "test_astar.py"],
        "checks": {
            "has_heapq": lambda files: any("heapq" in c for c in files.values()),
            "has_heuristic": lambda files: any("heuristic" in c.lower() or "manhattan" in c.lower() for c in files.values()),
            "has_diagonal": lambda files: any("diagonal" in c.lower() or "8" in c for c in files.values()),
            "has_path_return": lambda files: any("path" in c.lower() and "return" in c for c in files.values()),
            "has_obstacle_handling": lambda files: any("obstacle" in c.lower() or "wall" in c.lower() or "blocked" in c.lower() for c in files.values()),
            "has_tests": lambda files: any("test_" in name for name in files),
            "has_no_path_case": lambda files: any("no path" in c.lower() or "none" in c.lower() or "no solution" in c.lower() for c in files.values()),
        },
    },
}
