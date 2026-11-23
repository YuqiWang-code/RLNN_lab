"""Adapt the DISCS utility imports for the standalone MIS evaluator."""
from pathlib import Path
import sys


path = Path(sys.argv[1]) / "discs/common/utils.py"
source = path.read_text(encoding="utf-8")
source = source.replace("from clu import metric_writers\n", "")
source = source.replace("from clu.metric_writers.summary_writer import SummaryWriter\n", "")
source = source.replace("from discs.graph_loader import graph_gen\n", "")
source = source.replace('  """Setup logging and writer."""\n',
                        '  """Setup logging and writer."""\n'
                        '  from clu import metric_writers\n'
                        '  from clu.metric_writers.summary_writer import SummaryWriter\n')
source = source.replace("def get_datagen(config):\n",
                        "def get_datagen(config):\n  from discs.graph_loader import graph_gen\n")
path.write_text(source.replace("jax.tree_map(", "jax.tree.map("), encoding="utf-8")
