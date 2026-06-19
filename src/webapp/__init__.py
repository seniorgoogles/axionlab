"""Local control web app (FastAPI) for axionlab.

Start/stop training & QAT, trigger export + FINN build, watch live logs, and
browse past runs from the tracking DB(s). Phase 1 = local execution; remote
worker agents (distributed) build on the same JobManager interface.
"""
