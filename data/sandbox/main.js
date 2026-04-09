// Offline-friendly Pyodide sandbox CLI that runs without remote fetches.
// The original TypeScript entrypoint depends on Deno's JSR registry.
// This version only relies on local files so it can run in an air-gapped setup.
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { join as joinPath, resolve as resolvePath, isAbsolute } from "node:path";

const require = createRequire(import.meta.url);
if (!globalThis.require) {
  globalThis.require = require;
}

const resolvedFilename = fileURLToPath(import.meta.url);
const resolvedDirname = fileURLToPath(new URL(".", import.meta.url));
if (!globalThis.__filename) {
  globalThis.__filename = resolvedFilename;
}
if (!globalThis.__dirname) {
  globalThis.__dirname = resolvedDirname;
}

const pkgVersion = "0.0.7";

function ensureTrailingSlash(path) {
  return path.endsWith("/") ? path : `${path}/`;
}

function toPosixPath(path) {
  return path.replace(/\\/g, "/");
}

function ensureDirExists(path, label) {
  try {
    const info = Deno.statSync(path);
    if (!info.isDirectory) {
      throw new Error(`${label} is not a directory: ${path}`);
    }
  } catch (error) {
    console.error(
      `Expected ${label} at ${path}. Set SANDBOX_BASE to the bundle root or a directory containing the Pyodide assets.`,
    );
    console.error(error instanceof Error ? error.message : String(error));
    Deno.exit(1);
  }
}

function ensureFileExists(path, label) {
  try {
    const info = Deno.statSync(path);
    if (!info.isFile) {
      throw new Error(`${label} is not a file: ${path}`);
    }
  } catch (error) {
    console.error(`Could not find required file ${label} at ${path}.`);
    console.error(error instanceof Error ? error.message : String(error));
    Deno.exit(1);
  }
}

function canonicalizePackageName(name) {
  return name.toLowerCase().replace(/[-_.]+/g, "-");
}

function escapeForDoubleQuotedString(value) {
  return value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function buildLocalWheelUriMap(pyodideDir) {
  const mapping = {};
  for (const entry of Deno.readDirSync(pyodideDir)) {
    if (!entry.isFile || !entry.name.endsWith(".whl")) {
      continue;
    }
    const dist = entry.name.split("-")[0];
    const key = canonicalizePackageName(dist);
    const filePath = joinPath(pyodideDir, entry.name);
    const uri = pathToFileURL(filePath).href;
    if (!mapping[key]) {
      mapping[key] = [];
    }
    mapping[key].push(uri);
  }
  for (const key of Object.keys(mapping)) {
    mapping[key].sort((a, b) => a.length - b.length);
  }
  return mapping;
}

const defaultBasePath = resolvePath(
  fileURLToPath(new URL("../../", import.meta.url)),
);
const envBase = Deno.env.get("SANDBOX_BASE");
const baseFsPath = resolvePath(envBase ?? defaultBasePath);

ensureDirExists(
  baseFsPath,
  envBase ? "SANDBOX_BASE directory" : "default sandbox base directory",
);

const pyodideDir = joinPath(baseFsPath, "pyodide");
ensureDirExists(pyodideDir, "pyodide directory");

const pyodideModulePath = joinPath(pyodideDir, "pyodide.mjs");
ensureFileExists(pyodideModulePath, "pyodide.mjs");

const tempRootDir = joinPath(baseFsPath, "tmp", "pyodide_worker_runner");
await Deno.mkdir(tempRootDir, { recursive: true });
const tempDirFs = await Deno.makeTempDir({
  dir: tempRootDir,
  prefix: "run-",
});

const runtimePaths = {
  baseFsPath,
  pyodideDir,
  pyodideDirPosix: toPosixPath(pyodideDir),
  tempDirFs,
  tempDirPosix: ensureTrailingSlash(toPosixPath(tempDirFs)),
};

const localWheelUriMap = buildLocalWheelUriMap(runtimePaths.pyodideDir);

const { loadPyodide } = await import(pathToFileURL(pyodideModulePath).href);

const prepareEnvCodeTemplate = `
import datetime
import importlib
import json
import re
import sys
from typing import Union, TypedDict, List, Any, Callable, Literal

try:
    from pyodide.code import find_imports  # noqa
except ImportError:
    from pyodide import find_imports  # noqa

import pyodide_js  # noqa

sys.setrecursionlimit(400)

LOCAL_WHEEL_URIS = json.loads("__LOCAL_WHEEL_URIS__")


def _canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _find_wheel(package: str) -> str | None:
    normalized = _canonicalize(package)
    entries = LOCAL_WHEEL_URIS.get(normalized, [])
    if not entries:
        return None
    return sorted(entries, key=len)[0]


async def _quiet_load_package(package: str) -> None:
    options = dict(
        messageCallback=lambda *_: None,
        errorCallback=lambda *_: None,
    )
    await pyodide_js.loadPackage(package, **options)


class InstallEntry(TypedDict):
    module: str
    package: str


def _resolve_package_name(module: str, mapping: dict[str, str]) -> str:
    package_name = mapping.get(module)
    if package_name is not None:
        return package_name
    if "." in module:
        parts = module.split(".")
        for end in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:end])
            package_name = mapping.get(candidate)
            if package_name is not None:
                return package_name
        return parts[0]
    return module


def find_imports_to_install(imports: list[str]) -> list[InstallEntry]:
    \"""
    Given a list of module names being imported, return a list of dicts
    representing the packages that need to be installed to import those modules.
    The returned list will only contain modules that aren't already installed.
    Each returned dict has the following keys:
      - module: the name of the module being imported
      - package: the name of the package that needs to be installed
    \"""
    try:
        to_package_name = pyodide_js._module._import_name_to_package_name.to_py()
    except AttributeError:
        to_package_name = pyodide_js._api._import_name_to_package_name.to_py()

    to_install: list[InstallEntry] = []
    for module in imports:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            package_name = _resolve_package_name(module, to_package_name)
            to_install.append(
                dict(
                    module=module,
                    package=package_name,
                )
            )
            fallback_package = module.split(".")[0]
            if package_name != fallback_package:
                to_install.append(
                    dict(
                        module=module,
                        package=fallback_package,
                    )
                )
    unique: list[InstallEntry] = []
    seen = set()
    for entry in to_install:
        package = entry["package"]
        if package in seen:
            continue
        seen.add(package)
        unique.append(entry)
    return unique


async def install_imports(
    source_code_or_imports: Union[str, list[str]],
    additional_packages: list[str] = [],
    message_callback: Callable[
          [
              Literal[
                "failed",
              ],
              Union[InstallEntry, list[InstallEntry]],
          ],
          None,
      ] = lambda event_type, data: None,
) -> List[InstallEntry]:
    if isinstance(source_code_or_imports, str):
        try:
            imports: list[str] = find_imports(source_code_or_imports)
        except SyntaxError:
            return
    else:
        imports: list[str] = source_code_or_imports

    to_install = find_imports_to_install(imports)
    # Merge with additional packages
    for package in additional_packages:
        if package not in to_install:
            to_install.append(dict(module=package, package=package))

    if to_install:
        try:
            import micropip  # noqa
        except ModuleNotFoundError:
            await _quiet_load_package("micropip")
            import micropip  # noqa

        for entry in to_install:
            package = entry["package"]
            try:
                await _quiet_load_package(package)
                continue
            except Exception:
                pass
            local_wheel = _find_wheel(package)
            if local_wheel is not None:
                try:
                    await micropip.install(local_wheel)
                    continue
                except Exception:
                    message_callback("failed", str(local_wheel))
                    break
            try:
                await micropip.install(package)
            except Exception:
                message_callback("failed", package)
                break # Fail fast
    return to_install


def load_session_bytes(session_bytes: bytes) -> list[str]:
    """Load the session module."""
    import dill
    import io

    buffer = io.BytesIO(session_bytes.to_py())
    dill.session.load_session(filename=buffer)


def dump_session_bytes() -> bytes:
    """Dump the session module."""
    import dill
    import io

    buffer = io.BytesIO()
    dill.session.dump_session(filename=buffer)
    return buffer.getvalue()


def robust_serialize(obj):
    """Recursively converts an arbitrary Python object into a JSON-serializable structure.

    The function handles:
      - Primitives: str, int, float, bool, None are returned as is.
      - Lists and tuples: Each element is recursively processed.
      - Dictionaries: Keys are converted to strings (if needed) and values are recursively processed.
      - Sets: Converted to lists.
      - Date and datetime objects: Converted to their ISO format strings.
      - For unsupported/unknown objects, a dictionary containing a 'type'
        indicator and the object's repr is returned.
    """
    # Base case: primitives that are already JSON-serializable
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Process lists or tuples recursively.
    if isinstance(obj, (list, tuple)):
        return [robust_serialize(item) for item in obj]

    # Process dictionaries.
    if isinstance(obj, dict):
        # Convert keys to strings if necessary and process values recursively.
        return {str(key): robust_serialize(value) for key, value in obj.items()}

    # Process sets by converting them to lists.
    if isinstance(obj, (set, frozenset)):
        return [robust_serialize(item) for item in obj]

    # Process known datetime objects.
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()

    # Fallback: for objects that are not directly serializable,
    # return a dictionary with type indicator and repr.
    return {"type": "not_serializable", "repr": repr(obj)}


def dumps(result: Any) -> str:
    """Get the result of the session."""
    result = robust_serialize(result)
    return json.dumps(result)
`;

const wheelUriJson = JSON.stringify(localWheelUriMap);
const prepareEnvCode = prepareEnvCodeTemplate.replace(
  "__LOCAL_WHEEL_URIS__",
  escapeForDoubleQuotedString(wheelUriJson),
);

function parseBoolean(value, fallback) {
  if (value === undefined) {
    return true;
  }
  const normalized = value.toLowerCase();
  if (["true", "1", "yes", "on"].includes(normalized)) {
    return true;
  }
  if (["false", "0", "no", "off"].includes(normalized)) {
    return false;
  }
  return fallback;
}

function parseCliArgs(args) {
  const flags = {
    help: false,
    version: false,
    stateful: false,
  };

  const takeValue = (index, flag) => {
    if (index + 1 >= args.length) {
      throw new Error(`Missing value for ${flag}`);
    }
    return [args[index + 1], index + 1];
  };

  let i = 0;
  while (i < args.length) {
    const arg = args[i];

    if (arg.startsWith("--")) {
      const [name, value] = arg.split("=", 2);
      switch (name) {
        case "--code": {
          if (value !== undefined) {
            flags.code = value;
          } else {
            const [next, newIndex] = takeValue(i, "--code");
            flags.code = next;
            i = newIndex;
          }
          break;
        }
        case "--file": {
          if (value !== undefined) {
            flags.file = value;
          } else {
            const [next, newIndex] = takeValue(i, "--file");
            flags.file = next;
            i = newIndex;
          }
          break;
        }
        case "--session-bytes": {
          if (value !== undefined) {
            flags["session-bytes"] = value;
          } else {
            const [next, newIndex] = takeValue(i, "--session-bytes");
            flags["session-bytes"] = next;
            i = newIndex;
          }
          break;
        }
        case "--session-metadata": {
          if (value !== undefined) {
            flags["session-metadata"] = value;
          } else {
            const [next, newIndex] = takeValue(i, "--session-metadata");
            flags["session-metadata"] = next;
            i = newIndex;
          }
          break;
        }
        case "--stateful": {
          flags.stateful = parseBoolean(value, true);
          break;
        }
        case "--help": {
          flags.help = true;
          break;
        }
        case "--version": {
          flags.version = true;
          break;
        }
        default:
          throw new Error(`Unknown flag: ${name}`);
      }
      i += 1;
      continue;
    }

    if (arg.startsWith("-") && arg.length > 1) {
      switch (arg) {
        case "-c": {
          const [next, newIndex] = takeValue(i, "-c");
          flags.code = next;
          i = newIndex + 1;
          continue;
        }
        case "-f": {
          const [next, newIndex] = takeValue(i, "-f");
          flags.file = next;
          i = newIndex + 1;
          continue;
        }
        case "-b": {
          const [next, newIndex] = takeValue(i, "-b");
          flags["session-bytes"] = next;
          i = newIndex + 1;
          continue;
        }
        case "-m": {
          const [next, newIndex] = takeValue(i, "-m");
          flags["session-metadata"] = next;
          i = newIndex + 1;
          continue;
        }
        case "-s": {
          flags.stateful = true;
          i += 1;
          continue;
        }
        case "-h": {
          flags.help = true;
          i += 1;
          continue;
        }
        case "-V": {
          flags.version = true;
          i += 1;
          continue;
        }
        default:
          throw new Error(`Unknown flag: ${arg}`);
      }
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  return flags;
}

async function initPyodide(pyodide) {
  const dirPath = runtimePaths.tempDirPosix;
  const dirLiteral = JSON.stringify(dirPath);
  pyodide.runPython(
    `import pathlib; pathlib.Path(${dirLiteral}).mkdir(parents=True, exist_ok=True)`,
  );
  const sys = pyodide.pyimport("sys");
  const pathlib = pyodide.pyimport("pathlib");
  sys.path.append(dirPath);
  pathlib.Path(dirPath).joinpath("prepare_env.py").write_text(prepareEnvCode);
  pathlib.destroy?.();
  sys.destroy?.();
}

async function runPython(pythonCode, options) {
  const output = [];
  const errOutput = [];
  const originalLog = console.log;
  console.log = () => {};

  try {
    const pyodide = await loadPyodide({
      stdout: (msg) => output.push(msg),
      stderr: (msg) => errOutput.push(msg),
    });
    await pyodide.loadPackage(["micropip"], {
      messageCallback: () => {},
      errorCallback: (msg) => {
        output.push(`install error: ${msg}`);
      },
    });
    await initPyodide(pyodide);

    let sessionMetadata;
    if (options.sessionMetadata) {
      sessionMetadata = JSON.parse(options.sessionMetadata);
    } else {
      sessionMetadata = {
        created: new Date().toISOString(),
        lastModified: new Date().toISOString(),
        packages: [],
      };
    }
    let sessionData = null;

    if (options.sessionBytes && !options.sessionMetadata) {
      console.error("sessionMetadata is required when providing sessionBytes");
      return {
        success: false,
        error: "sessionMetadata is required when providing sessionBytes",
      };
    }

    const prepareEnv = pyodide.pyimport("prepare_env");
    const defaultPackages = options.stateful ? ["dill"] : [];
    const additionalPackagesToInstall = options.sessionBytes
      ? [...new Set([...defaultPackages, ...sessionMetadata.packages])]
      : defaultPackages;

    const installErrors = [];

    const installedPackages = await prepareEnv.install_imports(
      pythonCode,
      additionalPackagesToInstall,
      (eventType, data) => {
        if (eventType === "failed") {
          installErrors.push(data);
        }
      },
    );

    if (installErrors.length > 0) {
      console.log = originalLog;
      return {
        success: false,
        error:
          `Failed to install required Python packages: ${installErrors.join(", ")}. ` +
          `This is likely because these packages are not available in the Pyodide environment. ` +
          `Pyodide is a Python runtime that runs in the browser and has a limited set of ` +
          `pre-built packages. You may need to use alternative packages that are compatible ` +
          `with Pyodide.`,
      };
    }

    if (options.sessionBytes) {
      sessionData = Uint8Array.from(JSON.parse(options.sessionBytes));
      await prepareEnv.load_session_bytes(sessionData);
    }

    const packages = installedPackages.map((pkg) => pkg.get("package"));

    console.log = originalLog;
    const rawValue = await pyodide.runPythonAsync(pythonCode);
    const jsonValue = await prepareEnv.dumps(rawValue);

    sessionMetadata.packages = [
      ...new Set([...sessionMetadata.packages, ...packages]),
    ];
    sessionMetadata.lastModified = new Date().toISOString();

    if (options.stateful) {
      sessionData = await prepareEnv.dump_session_bytes();
    }

    const result = {
      success: true,
      result: rawValue,
      jsonResult: jsonValue,
      stdout: output,
      stderr: errOutput,
      sessionMetadata,
    };
    if (options.stateful && sessionData) {
      result.sessionBytes = sessionData;
    }
    return result;
  } catch (error) {
    return {
      success: false,
      error: error.message,
      stdout: output,
      stderr: errOutput,
    };
  } finally {
    console.log = originalLog;
  }
}

function printHelp() {
  console.log(`
pyodide-sandbox ${pkgVersion}
Run Python code in a sandboxed environment using Pyodide

OPTIONS:
  -c, --code <code>            Python code to execute
  -f, --file <path>            Path to Python file to execute
  -s, --stateful <bool>        Use a stateful session
  -b, --session-bytes <bytes>  Session bytes
  -m, --session-metadata       Session metadata
  -h, --help                   Display help
  -V, --version                Display version
`);
}

async function main() {
  try {
    let flags;
    try {
      flags = parseCliArgs(Deno.args);
    } catch (error) {
      console.error(error.message);
      Deno.exit(1);
      return;
    }

    if (flags.help) {
      printHelp();
      return;
    }

    if (flags.version) {
      console.log(pkgVersion);
      return;
    }

    if (!flags.code && !flags.file) {
      console.error(
        "Error: You must provide Python code using either -c/--code or -f/--file option.\nUse --help for usage information.",
      );
      Deno.exit(1);
      return;
    }

    const options = {
      code: flags.code,
      file: flags.file,
      stateful: flags.stateful,
      sessionBytes: flags["session-bytes"],
      sessionMetadata: flags["session-metadata"],
    };

    let pythonCode = "";

    if (options.file) {
      try {
        const filePath = isAbsolute(options.file)
          ? options.file
          : resolvePath(Deno.cwd(), options.file);
        pythonCode = await Deno.readTextFile(filePath);
      } catch (error) {
        console.error(`Error reading file ${options.file}:`, error.message);
        Deno.exit(1);
        return;
      }
    } else if (options.code) {
      pythonCode = options.code;
    }

    const result = await runPython(pythonCode, {
      stateful: options.stateful,
      sessionBytes: options.sessionBytes,
      sessionMetadata: options.sessionMetadata,
    });

    const outputJson = {
      stdout: result.stdout?.join("") || null,
      stderr: result.success
        ? (result.stderr?.join("") || null)
        : result.error || null,
      result: result.success ? JSON.parse(result.jsonResult || "null") : null,
      success: result.success,
      sessionBytes: result.sessionBytes,
      sessionMetadata: result.sessionMetadata,
    };

    console.log(JSON.stringify(outputJson));

    if (!result.success) {
      Deno.exit(1);
    }
  } finally {
    try {
      await Deno.remove(runtimePaths.tempDirFs, { recursive: true });
    } catch (_error) {
      // Directory might already be gone; ignore.
    }
  }
}

if (import.meta.main) {
  main().catch((err) => {
    console.error("Unhandled error:", err);
    Deno.exit(1);
  });
}

export { runPython };
