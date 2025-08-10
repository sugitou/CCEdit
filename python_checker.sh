python - <<'PY'
import importlib.util, inspect, taming, os
print("taming spec:", importlib.util.find_spec("taming"))
print("taming path:", os.path.dirname(inspect.getfile(taming)))
PY