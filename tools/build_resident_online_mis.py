"""Build the pinned KaMIS OnlineMIS with a memory input and incumbent callback."""
import argparse
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reference_methods/KaMIS/source"))
    parser.add_argument("--cmake-build", type=Path, default=Path("tmp/kamis_build"))
    parser.add_argument("--output", type=Path, default=Path("tmp/v16_anytime_build/online_mis.so"))
    args = parser.parse_args()
    source = args.source.resolve()
    original = (source / "lib/mis/ils/online_ils.cpp").read_text()
    header = (Path(__file__).resolve().parents[1] / "models/online_mis_bridge.cpp").read_text()
    journal = "best_solution = perm->get_solution_size() + perm->get_folded_vertices();"
    if original.count(journal) != 3:
        raise ValueError("Expected initialization and both improvement sites of pinned OnlineMIS")
    original = original.replace(journal, journal + "\n      publish(G, perm);")
    initial = "  local.direct_improvement(G, marked_degree, degree_limit);"
    if original.count(initial) != 1:
        raise ValueError("Expected the first feasible greedy solution in pinned OnlineMIS")
    original = original.replace(initial, "  publish(G, perm);\n" + initial)
    original = original.replace("if (t.elapsed() > config.time_limit) break;",
                                "if (Clock::now() >= stop_search) break;")
    original = original.replace("  mis_log::instance()->restart_online_timer();", "")
    original = original.replace("mis_log::instance()->set_best_size_online(config, best_solution);", "")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    instrumented = args.output.with_name("resident_online_ils.cpp")
    instrumented.write_text(header + "\n" + original)
    flags_file = args.cmake_build / "CMakeFiles/online_mis.dir/flags.make"
    flags = next(line.partition(" = ")[2] for line in flags_file.read_text().splitlines()
                 if line.startswith("CXX_INCLUDES = "))
    units = [
        "lib/mis/ils/local_search_online.cpp", "lib/mis/initial_mis/greedy_mis_online.cpp",
        "lib/mis/initial_mis/initial_mis.cpp", "lib/data_structure/mis_permutation_online.cpp",
        "lib/data_structure/candidate_list.cpp",
        "lib/data_structure/operation_log.cpp", "lib/data_structure/priority_queues/bucket_array.cpp",
        "extern/KaHIP/lib/tools/random_functions.cpp",
    ]
    subprocess.run(["g++", "-std=c++11", "-O3", "-march=native", "-funroll-loops", "-DNDEBUG",
                    "-fPIC", "-shared", "-fopenmp", "-Wl,-z,defs", "-DOMIS", *shlex.split(flags),
                    "-I" + str(source / "lib/mis/ils"), str(instrumented),
                    *(str(source / f) for f in units), "-o", str(args.output)], check=True)


if __name__ == "__main__":
    main()
