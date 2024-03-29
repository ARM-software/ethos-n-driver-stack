#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2018-2023 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#

import os
import re
from collections import namedtuple
import SCons.Script
import SCons.Variables.PathVariable as PathVariable


class Variant(namedtuple("Variant", ("ces", "ogs", "emcs", "ple_lanes"))):
    pass


variants = [
    Variant(ces=2, ogs=4, emcs=4, ple_lanes=2),
    Variant(ces=4, ogs=4, emcs=4, ple_lanes=2),
    Variant(ces=8, ogs=2, emcs=2, ple_lanes=2),
    Variant(ces=8, ogs=4, emcs=2, ple_lanes=2),
]

# Lookup from PLE parameter names to a short *unique* code to use to generate unique short identifiers
SHORT_PARAMS_LOOKUP = {
    "operation": "",  # Implicit from the value, so no prefix needed
    "block_width": "bw",
    "block_height": "bh",
    "block_multiplier": "bm",
    "datatype": "",  # Implicit from the value, so no prefix needed
    "is_direction_x": "dx",
    "is_direction_y": "dy",
    "pooling_size": "ps",
}


# Converts a dict of PLE kernel parameter names and values into a string that uniquely identifies it.
# This should be safe to use as a filename or identifier. e.g.:
#    { "operation": "ADDITION", "block_width":"16", "datatype": "u8" } => ADDITION_bw16_u8
def get_string_from_kernel_params(params):
    unique_name = []
    for k, v in params.items():
        # Keep it short by using a short code for each option
        unique_name.append("{}{}".format(SHORT_PARAMS_LOOKUP[k], v))
    # Remove invalid filename characters
    unique_name = [re.sub(r'[/\\?%*:|"<>,]', "", str(x)) for x in unique_name]
    return "_".join(unique_name)


# Similar to the above, but also includes the variant, producing something like:
#    V2442_ADDITION_RESCALE_bw16_bh16_bm1_u8
def get_string_from_variant_and_kernel_params(variant, params):
    params_string = get_string_from_kernel_params(params)
    # 'V' prefix (for variant) is important so that the identifier doesn't start with a number
    return f"V{variant.ces}{variant.ogs}{variant.emcs}{variant.ple_lanes}_{params_string}"


def create_variables():
    """
    Create a default scons var setup.

    This adds default parameters to the environment such as the options.py file,
    PATH and LD_LIBRARY_PATH are used for setting the respective environment variables,
    CPATH and LPATH are used for setting additional paths for C include files and library files respectively.

    Returns:
        (SCons.Script.Variables): The scons Variables pre-setup with common parameters
    """
    # First create a dummy env to parse just the 'options' parameter.
    # This allows the user to specify a file containing build options
    options_var = SCons.Script.Variables()
    options_var.Add("options", "Options for SConstruct e.g. debug=0", "options.py")
    env = SCons.Script.Environment(variables=options_var)

    # Use this parameter to create the real variables, pre-populating the values from the user provided
    # options file, and also the dev_options.py file, which may be pe present if this is a developer checkout.
    # Note we include the 'options' var again, so it appears in the help command
    var = SCons.Script.Variables([os.path.join("..", "internal", "dev_options.py"), env["options"]])
    var.AddVariables(
        ("options", "Options for SConstruct e.g. debug=0", "options.py"),
        ("PATH", "Prepend to the PATH environment variable"),
        ("LD_LIBRARY_PATH", "Prepend to the LD_LIBRARY_PATH environment variable"),
        ("CPATH", "Append to the C include path list the compiler uses"),
        ("LPATH", "Append to the library path list the compiler uses"),
        ("scons_extra", "Extra scons files to be loaded, separated by comma.", ""),
        ("scons_dev_optional", "Optional scons files to be loaded, separated by comma.", ""),
        PathVariable(
            "install_prefix",
            "Installation prefix",
            os.path.join(os.path.sep, "usr", "local"),
            PathVariable.PathAccept,
        ),
        PathVariable(
            "install_bin_dir",
            "Executables installation directory",
            os.path.join("$install_prefix", "bin"),
            PathVariable.PathAccept,
        ),
        PathVariable(
            "install_include_dir",
            "Header files installation directory",
            os.path.join("$install_prefix", "include"),
            PathVariable.PathAccept,
        ),
        PathVariable(
            "install_lib_dir",
            "Libraries installation directory",
            os.path.join("$install_prefix", "lib"),
            PathVariable.PathAccept,
        ),
    )
    return var


def load_extras(env):
    "Load any extra scons scripts, specified in scons_extra variable"
    scriptpath = env.get("scons_extra")
    if scriptpath:
        scripts = [s for s in scriptpath.split(",") if s]  # Ignore empty entries
        for s in scripts:
            env.SConscript(s, exports=["env"])


def load_optional(env):
    "Load any optional dev scons scripts, specified in scons_dev_optional variable"
    scriptpath = env.get("scons_dev_optional")
    if scriptpath:
        scripts = [s for s in scriptpath.split(",") if s]  # Ignore empty entries
        for s in scripts:
            opt = os.path.split(s)
            if "dev_optional_params.scons" in opt:
                root_dir = env.Dir("#").abspath
                path = os.path.join(root_dir, s)
                if not os.path.isfile(path):
                    return False
                env.SConscript(s, exports=["env"])
                return True

    return False


def add_env_var(env, variable):
    """
    Add a scons variable into the scons environment as an environment variable, but only if it has been set.

    Args:
        env (SCons.Environment): The scons environment to use
        variable          (str): The scons parameter to promote to an envvar
    """
    if variable in env:
        env["ENV"][variable] = env[variable]


def parse_default_vars(env):
    """
    Parse the default variables that are defined in the create_variables() function.

    Args:
        env (SCons.Environment): The scons environment to use
    """
    # Import the ARMLMD_LICENSE_FILE environment variable into scons.
    if "ARMLMD_LICENSE_FILE" in os.environ:
        env["ENV"]["ARMLMD_LICENSE_FILE"] = os.environ["ARMLMD_LICENSE_FILE"]
    # Allows colours to be used e.g. for errors
    if "TERM" in os.environ:
        env["ENV"]["TERM"] = os.environ["TERM"]
    # Allow processes launched by scons to detect the processor architecture.
    if "PROCESSOR_ARCHITECTURE" in os.environ:
        env["ENV"]["PROCESSOR_ARCHITECTURE"] = os.environ["PROCESSOR_ARCHITECTURE"]
    # Allow python processes launched by scons to honour PYTHONUNBUFFERED.
    # I think a better way of handling this would be to have each script set this itself
    # if that script is launching subprocesses, but I haven't found a good way of doing this.
    if "PYTHONUNBUFFERED" in os.environ:
        env["ENV"]["PYTHONUNBUFFERED"] = os.environ["PYTHONUNBUFFERED"]
    # Prepend to the PATH env additional paths to search
    if "PATH" in env:
        env.PrependENVPath("PATH", env["PATH"])
    # Prepend to the LD_LIBRARY_PATH env additional paths to search when executing through scons
    if "LD_LIBRARY_PATH" in env:
        env.PrependENVPath("LD_LIBRARY_PATH", env["LD_LIBRARY_PATH"])
    # Because these path arguments may be relative, they must be correctly interpreted as relative to the top-level
    # folder rather than the 'build' subdirectory, which is what scons would do if they were passed to the SConscript
    # as-is. Therefore we convert them to absolute paths here, where they will be interpreted correctly.
    if "CPATH" in env:
        env.AppendUnique(CPPPATH=[os.path.abspath(x) for x in env["CPATH"].split(os.pathsep)])
    if "LPATH" in env:
        env.AppendUnique(LIBPATH=[os.path.abspath(x) for x in env["LPATH"].split(os.pathsep)])
    # If the given path is not absolute, derive it from the current working directory.
    if not os.path.isabs(env["install_prefix"]):
        env.Replace(
            install_prefix=os.path.join(os.path.abspath(os.getcwd()), env["install_prefix"])
        )


def setup_common_env(env):
    """
    Setup the common SConstruct build environment with default values.

    Args:
        env (SCons.Environment): The scons environment to use
    """
    # Secure Development Lifecycle
    # The following is a set of security compilation flags required
    env.AppendUnique(CPPFLAGS=["-Wall", "-Wextra"])
    if env["werror"]:
        env.AppendUnique(CPPFLAGS=["-Werror"])
    # Increase the warning level for 'format' to 2, but disable the nonliteral case
    env.AppendUnique(CPPFLAGS=["-Wformat=2", "-Wno-format-nonliteral"])
    env.AppendUnique(
        CPPFLAGS=[
            "-Wctor-dtor-privacy",
            "-Woverloaded-virtual",
            "-Wsign-promo",
            "-Wstrict-overflow=2",
            "-Wswitch-default",
            "-Wlogical-op",
            "-Wnoexcept",
            "-Wstrict-null-sentinel",
            "-Wconversion",
        ]
    )
    # List of flags that should be set but currently fail
    # env.AppendUnique(CPPFLAGS=["-Weffc++"])
    # env.AppendUnique(CPPFLAGS=["-pedantic", "-fstack-protector-strong"])

    env.AppendUnique(CPPFLAGS=["-fPIC"])
    env.Replace(CXXSTD="c++14")
    env.AppendUnique(CXXFLAGS=["-std=$CXXSTD"])
    if env["debug"]:
        env.AppendUnique(CXXFLAGS=["-O0", "-g"])
    else:
        env.AppendUnique(CXXFLAGS=["-O3"])
    env.PrependUnique(CPPPATH=["include"])

    # Setup asserts, if asserts are explicitly set then set NDEBUG accordingly.
    # If asserts aren't explicitly set then enable asserts in debug and disable them in release.
    if env["asserts"] == "0" or (env["asserts"] == "debug" and not env["debug"]):
        env.AppendUnique(CPPDEFINES=["NDEBUG"])

    if env["logging"] == "1" or (env["logging"] == "debug" and env["debug"]):
        env.AppendUnique(CPPDEFINES=["ETHOSN_LOGGING"])

    if env.get("coverage", False):
        env.AppendUnique(CXXFLAGS=["--coverage", "-O0"])
        env.AppendUnique(LINKFLAGS=["--coverage"])
    # By enabling this flag, binary will use RUNPATH instead of RPATH
    env.AppendUnique(LINKFLAGS=["-Wl,--enable-new-dtags"])


def setup_sanitize(env):
    # Add sanitization flags
    if env["sanitize"]:
        flags = (
            "-fsanitize=address",
            "-fsanitize-address-use-after-scope",
            "-fsanitize=undefined",
            "-fsanitize=leak",
            "-fsanitize-recover=address",
        )
        env.AppendUnique(CXXFLAGS=flags)
        env.AppendUnique(CPPFLAGS=flags)
        env.AppendUnique(LINKFLAGS=flags)
        # if werror is enabled, the sanitizer throws an integer to uint16_t conversion
        # error at compile time and doesn't detect the runtime address/undefined errors
        env["CPPFLAGS"].remove("-Werror")


def remove_flags(flags_list, environment):
    for flag_to_remove in flags_list:
        if flag_to_remove in environment:
            environment.remove(flag_to_remove)


def setup_toolchain(env, toolchain):
    """
    Setup the scons toolchain using predefined toolchains.

    Args:
        env (SCons.Environment): The scons environment to use
        toolchain         (str): One of the following strings are accepted ('aarch64', 'armclang', 'native')
                                 Any other value defaults to the same as native
    """
    if toolchain == "aarch64":
        env.Replace(
            CC="aarch64-linux-gnu-gcc",
            CXX="aarch64-linux-gnu-g++",
            LINK="aarch64-linux-gnu-g++",
            AS="aarch64-linux-gnu-as",
            AR="aarch64-linux-gnu-ar",
            RANLIB="aarch64-linux-gnu-ranlib",
        )
    elif toolchain == "armclang":
        env.Replace(
            CC="armclang --target=arm-arm-none-eabi",
            CXX="armclang --target=arm-arm-none-eabi",
            LINK="armlink",
            AS="armclang --target=arm-arm-none-eabi",
            AR="armar",
            RANLIB="armar -s",
        )
        # List of flags armclang doesnt understand so should be removed from the inherited common set of flags
        if "CPPFLAGS" in env:
            flags = ["-Wlogical-op", "-Wnoexcept", "-Wstrict-null-sentinel"]
            remove_flags(flags, env["CPPFLAGS"])
        if "LINKFLAGS" in env:
            flags = ["-Wl,--enable-new-dtags"]
            remove_flags(flags, env["LINKFLAGS"])
    elif toolchain == "llvm-embedded":
        env.Replace(
            CC="clang --target=arm-arm-none-eabi",
            CXX="clang --target=arm-arm-none-eabi",
            LINK="ld.lld",
            AS="clang --target=arm-arm-none-eabi",
            AR="armar",
            RANLIB="armar -s",
        )

        # List of flags clang doesnt understand so should be removed from the inherited common set of flags
        if "CPPFLAGS" in env:
            flags = ["-Wlogical-op", "-Wnoexcept", "-Wstrict-null-sentinel"]
            remove_flags(flags, env["CPPFLAGS"])
        if "LINKFLAGS" in env:
            flags = ["-Wl,--enable-new-dtags"]
            remove_flags(flags, env["LINKFLAGS"])


def validate_dir(env, path, exception_type):
    """
    Validate a directory exists, raising a specific exception in the case it is not valid.

    This is used for validating directories that cannot use the scons PathVariable validation mechanism,
    i.e. it only needs to be valid if another scons variable is set.

    Args:
        env               (SCons.Environment): The scons environment to use
        path                            (str): The scons parameter to validate
        exception_type (exceptions.Exception): The exception type to throw in the event of an invalid path
    """
    if not os.path.isdir(env[path]):
        raise exception_type("\033[91mERROR: {} is not a valid directory.\033[0m".format(path))


def parse_int(env, variable, exception_type):
    """
    Validate a variable is an int, raising a specific exception in the case it is not.

    This is used for validating scons variables that should be of type int (or castable to that type).

    Args:
        env               (SCons.Environment): The scons environment to use
        variable                        (str): The scons parameter to validate
        exception_type (exceptions.Exception): The exception type to throw in the event of an invalid variable
    """
    try:
        env[variable] = int(env[variable])
    except:
        raise exception_type("\033[91mERROR: {} is not a valid value.\033[0m".format(variable))


def variable_exists(env, variable, exception_type):
    """
    Check that a variable exists, raising a specific exception in the case it is not.

    This is used for validating scons variables that must be set.

    Args:
        env               (SCons.Environment): The scons environment to use
        variable                        (str): The scons parameter to validate
        exception_type (exceptions.Exception): The exception type to throw in the event of an invalid variable
    """
    if variable not in env:
        raise exception_type(
            '\033[91mERROR: Missing required "{}" parameter.\033[0m'.format(variable)
        )


def abs_path(env, paths, relative_offset="."):
    """
    Convert path(s) to their absolute path equivalent.

    Args:
        env   (SCons.Environment): The scons environment to use
        paths (str or list/tuple): The scons parameter(s) to abspath
        relative_offset (str)    : When a relative path needs converting to absolute, it is interpreted relative to this
    """
    paths = paths if isinstance(paths, (list, tuple)) else [paths]
    for path in paths:
        try:
            if not os.path.isabs(env[path]):
                env[path] = env.Dir(os.path.join(relative_offset, env[path])).abspath
        except KeyError:
            continue


def abs_filepath(env, paths, relative_offset="."):
    """
    Convert file path(s) to their absolute file path equivalent.

    Args:
        env   (SCons.Environment): The scons environment to use
        paths (str or list/tuple): The scons parameter(s) to conviert into absolute file path
        relative_offset (str)    : When a relative path needs converting to absolute, it is interpreted relative to this
    """
    paths = paths if isinstance(paths, (list, tuple)) else [paths]
    for path in paths:
        try:
            if not os.path.isabs(env[path]):
                env[path] = env.File(os.path.join(relative_offset, env[path])).abspath
        except KeyError:
            continue
        except TypeError:
            # If path to file is not given, SCons assumes that value is '.', which is a directory
            continue


def get_build_type(env):
    return "debug" if env["debug"] else "release"


def get_build_dir(env, module_base, config=None):
    if config is None:
        config = "{}_{}".format(get_build_type(env), env["platform"])
    return env.Dir(os.path.join(module_base, env["build_dir"], config)).abspath


def get_driver_library_build_dir(env):
    config = "{}_{}_{}".format(get_build_type(env), env["platform"], env["target"])
    return get_build_dir(env, env["driver_library_dir"], config)


def get_support_library_build_dir(env):
    return get_build_dir(env, env["support_library_dir"])


def get_command_stream_build_dir(env):
    return get_build_dir(env, env["command_stream_dir"])


def get_utils_build_dir(env):
    return get_build_dir(env, env["utils_dir"])


def get_lib_xml_to_binary_build_dir(env):
    return get_build_dir(env, env["lib_xml_to_binary_dir"])


def get_system_tests_build_dir(env):
    return get_build_dir(env, env["system_tests_dir"])


def get_armnn_build_dir(env):
    return get_build_dir(env, env["armnn_dir"])


def get_single_elem(elems, msg_context):
    "If a single element exists in the list, returns that. If length is not 1, it throws error"
    if len(elems) != 1:
        raise SCons.Errors.UserError("{} : Elems List size needs to be 1".format(msg_context))
    return elems[0]


def root_dir():
    "Returns the root of driver_stack tree"
    return os.path.realpath(os.path.join(__file__, "..", ".."))


def add_padding(align):
    """
    Return a function that can be used in a Command target to pad a file to a multiple of align bytes in place.
    """

    def add_padding_fn(env, source, target):
        with open(target[0].path, "ab") as f:
            sz = f.tell()
            new_sz = ((sz + align - 1) // align) * align
            f.write(b"\x00" * (new_sz - sz))

    return add_padding_fn


def get_control_unit_build_dir(env, backend):
    build_type = get_build_type(env)
    if env["profiling"]:
        profiling = "_profiling"
    else:
        profiling = ""
    config = "{}_{}{}".format(build_type, backend, profiling)
    return get_build_dir(env, env["control_unit_dir"], config)


def setup_plelib_dependency(env):
    """
    Setup plelib dependency.

    Args:
        env (SCons.Environment): The scons environment to use
    """
    ple_include = os.path.join(get_ple_build_dir(env), "include")

    # Note we *prepend* so these take priority over CPATH command-line-arguments to avoid depending on
    # the install target where the install target is also provided via CPATH.
    env.PrependUnique(CPPPATH=[ple_include])


def get_ple_build_dir(env):
    config = "release"
    return get_build_dir(env, env["ple_dir"], config)
