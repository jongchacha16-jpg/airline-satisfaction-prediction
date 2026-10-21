import sys, nbformat
from nbclient import NotebookClient
p = sys.argv[1]
nb = nbformat.read(p, as_version=4)
NotebookClient(nb, timeout=7200, kernel_name="sesac-ai", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, p)
print("DONE", p)
