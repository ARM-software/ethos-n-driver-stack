#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2021,2023 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#

# Parses a .csv file of coprocessor instruction encodings (from the spec)
# into a database of instructions. This can be used in both directions -
# to export C++ function definitions which expose the SW interface to the HW instructions,
# and also to decode HW instructions into the more friendly SW description.

# At a high level, it works by building up a mapping between the SW and HW instructions.
# Each instruction has a list of arguments, for example the HW instruction has things like
# coprocessor register numbers (e.g. CRn) and opcodes (e.g. op1), whereas the SW instruction
# has more specific arguments like "swzsel". These arguments are often linked together, e.g.
# the hardware "op1" argument might correspond to the SW "swzsel" argument. More specifically,
# each invidivudal bit of each argument might be linked in this way (e.g. bit 3 of op1 corresponds
# to bit 0 of swzsel). This mapping between the bits fully describes the instruction and
# can be used to both decode and encode the instructions.

import os
import sys
import csv
import re
from collections import OrderedDict
import argparse


class Instruction(object):
    """
    A single HW or SW instruction, comprising one or more arguments which link to other arguments in the corresponding
    'other' (HW <=> SW) instruction.
    """

    def __init__(self, name):
        self._name = name
        self._arguments = OrderedDict()

    @property
    def name(self):
        return self._name

    def add_argument(self, name, bits):
        if name in self._arguments:
            # Setting the same argument more than once does happen for some instructions, as the link from SW <=> HW
            # is specified both ways (e.g. for SET_SWZSEL_REG_SEL, swzsel <=> opcodes)
            # Therefore we allow this, only as long as the new definition is exactly the same
            if self._arguments[name] != Argument(name, bits):
                raise Exception("This argument has already been set with a mismatching definition")
        else:
            self._arguments[name] = Argument(name, bits)

    def get_argument(self, name):
        return self._arguments[name]

    def get_argument_or_none(self, name):
        return self._arguments[name] if name in self._arguments else None

    def get_argument_and_add_if_not_exists(self, name):
        if name not in self._arguments:
            self._arguments[name] = Argument(name, [])
        return self.get_argument(name)

    def get_argument_names(self):
        return self._arguments.keys()

    def get_arguments(self):
        return self._arguments.values()

    def sort_arguments(self, order):
        # Note that arguments which are not provided in `order` will be left at the end of arguments list.
        for key in reversed(order):
            if key in self._arguments:
                self._arguments.move_to_end(key, last=False)

    def __repr__(self):
        # For better display in debugger
        return "{} {}".format(self._name, ", ".join([str(arg) for arg in self._arguments.values()]))


class HwInstruction(Instruction):
    def __init__(self, name, hw_instruction_type):
        super().__init__(name)
        self._hw_instruction_type = hw_instruction_type

    @property
    def hw_instruction_type(self):
        return self._hw_instruction_type


class Argument(object):
    """
    A single argument of either the HW or SW instruction, consisting of one or more bits which may be fixed values
    or linked to bits in the corresponding 'other' (HW <=> SW) instruction.
    """

    def __init__(self, name, bits):
        self._name = name
        self._bits = bits

    def set_or_add_bit(self, bit_idx, bit):
        while bit_idx >= len(self._bits):
            self._bits.append(None)
        if self._bits[bit_idx] is not None:
            # Setting the same bit more than once does happen for some instructions, as the link from SW <=> HW
            # is specified both ways (e.g. for SET_SWZSEL_REG_SEL, swzsel <=> opcodes)
            # Therefore we allow this, only as long as the new definition is exactly the same
            if self._bits[bit_idx] != bit:
                raise Exception("This bit has already been set with a mismatching definition!")
        self._bits[bit_idx] = bit

    @property
    def name(self):
        return self._name

    def get_num_bits(self):
        return len(self._bits)

    @property
    def bits(self):
        return self._bits

    def __repr__(self):
        # For better display in debugger
        return "{}=[{}]".format(self._name, ",".join([str(b) for b in self._bits]))

    def __eq__(self, other):
        # pylint: disable=protected-access
        if isinstance(other, Argument):
            return self._name == other._name and self._bits == other._bits
        return NotImplemented


class Bit:
    """
    A single bit in an Argument, which may be a fixed value (ConstantBit) or a link to a bit
    in the corresponding 'other' (HW <=> SW) instruction (LinkedBit).
    """

    @classmethod
    def is_constant(cls):
        return False

    @classmethod
    def is_linked(cls):
        return False


class ConstantBit(Bit):
    """
    A single bit in an Argument, which has a fixed value (0 or 1).
    """

    def __init__(self, value):
        self._value = value

    @classmethod
    def is_constant(cls):
        return True

    @property
    def value(self):
        return self._value

    def __repr__(self):
        # For better display in debugger
        return str(self._value)

    def __eq__(self, other):
        # pylint: disable=protected-access
        if isinstance(other, ConstantBit):
            return self._value == other._value
        return NotImplemented


class LinkedBit(Bit):
    """
    A single bit in an Argument, which is linked to a bit
    in the corresponding 'other' (HW <=> SW) instruction.
    """

    def __init__(self, linked_argument_name, linked_bit_idx):
        self._linked_argument_name = linked_argument_name
        self._linked_bit_idx = linked_bit_idx

    @classmethod
    def is_linked(cls):
        return True

    @property
    def linked_argument_name(self):
        return self._linked_argument_name

    @property
    def linked_bit_idx(self):
        return self._linked_bit_idx

    def __repr__(self):
        # For better display in debugger
        return "{}[{}]".format(self._linked_argument_name, self._linked_bit_idx)

    def __eq__(self, other):
        # pylint: disable=protected-access
        if isinstance(other, LinkedBit):
            return (
                self._linked_argument_name == other._linked_argument_name
                and self._linked_bit_idx == other._linked_bit_idx
            )
        return NotImplemented


class CoprocessorInstructionDatabase(object):
    def __init__(self):
        self._instructions = []
        self._descriptions = {}
        self._timings = {}

        self._verilog_constant_regex = re.compile(
            r"^(?P<numbits>\d+)'(?P<base>h|b|d|o)(?P<value>[0-9A-F]+)$"
        )
        self._verilog_variable_regex = re.compile(r"^(?P<name>\w+)\[(?P<msb>\d)(:(?P<lsb>\d))?\]$")

    def load(self, encodings_input_file, descriptions_input_file, timings_input_file):
        if descriptions_input_file:
            with open(descriptions_input_file, "rt", encoding="utf-8") as file:
                for row in csv.reader(file):
                    # Skip first two cols (Name and Old name) and filter out empty cells
                    self._descriptions[row[0]] = [x for x in row[2:] if x]

        if timings_input_file:
            with open(timings_input_file, "rt", encoding="utf-8") as file:
                for row in csv.reader(file):
                    self._timings[row[0]] = row

        with open(encodings_input_file, "rt", encoding="utf-8") as file:
            csv_reader = csv.reader(file)

            # Read header row to find out which columns we are interested in
            header = next(csv_reader)
            first_column = header.index("Software")
            last_column = header.index("Software Temp") - 1

            # Skip sub-headings
            sub_headings = next(csv_reader)
            field_headings = sub_headings[first_column:last_column]

            # Read and process all data rows
            for row in csv_reader:
                # Split into fields and extract the ones we care about
                fields = row[first_column:last_column]

                try:
                    self._process_row(fields, field_headings)
                except NotImplementedError as e:
                    print(
                        "Error processing row for {} - {}".format(fields[0], e),
                        file=sys.stderr,
                    )

    def _process_row(self, fields, field_names):
        # Extract the data we need from the row.
        hw_name = fields[0]
        # Some rows are blank (or only used as a sub-headings), so ignore these
        if not hw_name:
            return

        # Some columns have fixed names:
        hw_instruction_type = fields[field_names.index("COPRO_INST")]
        hw_opcode_1 = fields[field_names.index("Opcode1")]
        hw_opcode_2 = fields[field_names.index("Opcode2")]
        # The rest of the columns after Opcode2 are named after the SW arguments for that instruction
        sw_args = OrderedDict(
            (field_names[i], fields[i])
            for i in range(field_names.index("Opcode2") + 1, len(field_names))
            if field_names[i] and fields[i]
        )

        # Construct the corresponding HW and SW instructions - we will then fill these in
        hw_instruction = HwInstruction(hw_name, hw_instruction_type)
        sw_instruction = Instruction(self._hw_to_sw_name(hw_name))

        # MCR and MCRR instructions do not explicitly mention the Arm register parameter(s) in the spreadsheet
        # so we have to assume there is a 1:1 mapping between SW arguments and HW arguments
        if hw_instruction_type in ("MCR", "MCR2"):
            self._add_argument(
                hw_instruction,
                "Rt",
                self._parse_verilog_bitfield("Rt[4:0]"),
                sw_instruction,
            )
        elif hw_instruction_type in ("MCRR", "MCRR2"):
            self._add_argument(
                hw_instruction,
                "Rt",
                self._parse_verilog_bitfield("Rt[4:0]"),
                sw_instruction,
            )
            self._add_argument(
                hw_instruction,
                "Rt2",
                self._parse_verilog_bitfield("Rt2[4:0]"),
                sw_instruction,
            )

        # Parse the opcode columns. These are normally simple constants but some are expressions which
        # reference extra software arguments (which don't have a corresponding column)
        if hw_opcode_1:
            self._add_argument(
                hw_instruction,
                "op1",
                self._parse_verilog_bitfield(hw_opcode_1),
                sw_instruction,
            )
        if hw_opcode_2:
            self._add_argument(
                hw_instruction,
                "op2",
                self._parse_verilog_bitfield(hw_opcode_2),
                sw_instruction,
            )

        # Parse the software args columns
        for sw_arg_name, sw_arg_expr in sw_args.items():
            if sw_arg_name == "Immediate":
                # The 'Immediate' column is different to the others -
                # the cell value contains an expression which assigns a hardware register to a SW instruction argument
                sw_arg_name, bits = self._parse_immediate_field(sw_arg_expr)
            else:
                # All other arguments contain a single verilog bitfield
                bits = self._parse_verilog_bitfield(sw_arg_expr)

            self._add_argument(sw_instruction, sw_arg_name, bits, hw_instruction)

        # Sort the arguments into a sensible order for consistency (e.g. for the generated C++ code and for tests)
        hw_instruction.sort_arguments(["CPNUM", "op1", "op2", "CRd", "CRn", "CRm"])
        sw_instruction.sort_arguments(["Dest", "Src0", "Src1", "Rt"])

        self._instructions.append((hw_instruction, sw_instruction))

    @staticmethod
    def _hw_to_sw_name(hw_name):
        return "ve_" + hw_name.replace(".", "_").lower()

    def _parse_verilog_bitfield(self, s):
        """
        Parses a string of the form "2'b1,swzsel[3]"
        And converts it into a list of Bits
        """
        parts = s.split(",")
        bits = []
        for part in parts[::-1]:  # Note we loop in reverse to get LSBs first
            bits.extend(self._parse_verilog_bitfield_part(part))
        return bits

    def _parse_verilog_bitfield_part(self, s):
        """
        Parses a string of the form: "2'b1" or "swzsel[3]"
        And converts it into a list of Bits
        """
        s = s.strip()
        verilog_constant_match = self._verilog_constant_regex.match(s)
        verilog_variable_match = self._verilog_variable_regex.match(s)
        if verilog_constant_match:
            num_bits = int(verilog_constant_match.group("numbits"))
            base = {"h": 16, "b": 2, "d": 10, "o": 8}[verilog_constant_match.group("base")]
            value = int(verilog_constant_match.group("value"), base)
            binary = bin(value)[2:].zfill(num_bits)[::-1]  # Note we reverse to get LSB first
            return [ConstantBit(int(x)) for x in binary]
        elif verilog_variable_match:
            name = verilog_variable_match.group("name")
            msb = int(verilog_variable_match.group("msb"))
            lsb = int(
                verilog_variable_match.group("lsb") or msb
            )  # foo[3] is equivalent to foo[3:3]
            return [LinkedBit(name, i) for i in range(lsb, msb + 1)]
        else:
            raise NotImplementedError("Unexpected format for verilog bitfield part: {}".format(s))

    def _parse_immediate_field(self, expr):
        """
        Parses a string of the form: "Imm5[4:0]=CPNUM[0], CRm[3:0]"
        And returns a tuple of ("Imm5", <List of Bits from RHS>)
        """
        sw_part, hw_part = expr.split("=")
        sw_bits = self._parse_verilog_bitfield(sw_part)
        hw_bits = self._parse_verilog_bitfield(hw_part)
        if len(sw_bits) != len(hw_bits):
            raise ValueError(
                "Immediate field has different number of bits either side of equals sign"
            )
        if not all(
            map(
                lambda bit_enum: bit_enum[1].linked_argument_name == sw_bits[0].linked_argument_name
                and bit_enum[1].linked_bit_idx == bit_enum[0],
                enumerate(sw_bits),
            )
        ):
            raise ValueError("Immediate field LHS should be a single argument")

        return (sw_bits[0].linked_argument_name, hw_bits)

    @staticmethod
    def _add_argument(instruction, arg_name, bits, other_instruction):
        """
        Adds the given list of bits as a named argument to the given instruction, and also adds the reverse
        links from the other_instruction.
        """
        instruction.add_argument(arg_name, bits)
        for bit_idx, b in enumerate(bits):
            if b is not None and b.is_linked():
                # The other instruction might not yet have an argument defined, if this is the first time we've seen it,
                # so add one if it doesn't exist. Same goes for bits within that argument.
                reverse_linked_bit = LinkedBit(arg_name, bit_idx)
                other_instruction.get_argument_and_add_if_not_exists(
                    b.linked_argument_name
                ).set_or_add_bit(b.linked_bit_idx, reverse_linked_bit)

    def _get_formatted_description(self, instruction_name):
        # Get all the description cells
        cells = self._descriptions.get(instruction_name, ["NO DESCRIPTION AVAILABLE"])
        # Each cell in the array may itself be a multi-line string, so expand these to separate elements in the array,
        # trim trailing whitespace on each line and prepend with comment
        lines = [("// " + line).rstrip() for cell in cells for line in cell.split("\n")]
        # Combine into a single string
        return "\n".join(lines) + "\n"

    def _get_formatted_timing(self, instruction_name):
        # Get all the description cells
        cells = self._timings.get(instruction_name, [])
        c_name = self._hw_to_sw_name(instruction_name).upper()
        lines = ["struct " + c_name.replace("VE_", ""), "{"]
        for i, cell in enumerate(cells[1:]):
            if not cell or cell == "N/A":
                # Timing not available for this instruction
                continue
            stage = self._timings["\ufeffInstruction"][i + 1].upper()
            # Demux on stages
            if stage == "RF READ":
                veStageName = "OP_READ"
            elif stage == "FLAG READ":
                veStageName = "FLAG_READ"
            elif stage == "FLAG WRITE":
                veStageName = "FLAG_WRITE"
            elif stage == "EXECUTE":
                veStageName = "EXECUTE"
            elif stage.startswith("RF, ACCU"):
                veStageName = "WRITE_BACK"
            elif stage == "COMPLETE":
                # Not interesting as the MCU always
                # spends one cycle issuing the instruction
                # i.e. the FETCH stage is done by the MCU
                continue
            elif stage.startswith("PIPELINE"):
                # Not a stage but useful information
                # anyway
                veStageName = "PIPELINE"
            else:
                raise NotImplementedError("Stage {} not implemented".format(stage))
            assert cell.isdigit(), "Value {} for {} is not int".format(cell, instruction_name)
            lines.append(
                "    constexpr static unsigned int {stage: <{fill}} = {val};".format(
                    stage=veStageName, fill=len("WRITE_BACK"), val=cell
                )
            )
        # Combine into a single string
        return "\n".join(lines) + "\n};\n"

    def export_cpp_functions(self, outputFile):
        with open(outputFile, "wt") as output:
            output.write(
                """//
// Copyright:
// ----------------------------------------------------------------------------
// This confidential and proprietary software may be used only as authorized
// by a licensing agreement from Arm Limited.
//      (C) COPYRIGHT 2021 Arm Limited
// The entire notice above must be reproduced on all authorized copies and
// copies may only be made to the extent permitted by a licensing agreement
// from Arm Limited.
// ----------------------------------------------------------------------------
//
// This file was automatically generated by {}

#pragma once

#include "ethosn_ple/utils.h"

namespace    // Internal linkage
{{
namespace VE_TIMING
{{
""".format(
                    os.path.basename(__file__)
                )
            )

            for (hw_instruction, sw_instruction) in self._instructions:
                s = self._get_formatted_timing(hw_instruction.name)
                output.write(s + "\n")
            output.write("}    // namespace VE_TIMING\n\n")

            for (hw_instruction, sw_instruction) in self._instructions:
                code = self._export_cpp_function(hw_instruction, sw_instruction)
                output.write(code + "\n")

            output.write("}    // namespace\n")

    def _export_cpp_function(self, hw_instruction, sw_instruction):
        code = self._get_formatted_description(hw_instruction.name).replace(
            hw_instruction.name, sw_instruction.name
        )
        # Split SW arguments into runtime and template args
        runtime_arg_names = [x for x in sw_instruction.get_argument_names() if x.startswith("R")]
        template_arg_names = [
            x for x in sw_instruction.get_argument_names() if not x.startswith("R")
        ]
        extra_template_arg_names = ["post_cc = 0"]
        template_arg_names_str = ", ".join(
            "unsigned int {}".format(x) for x in template_arg_names + extra_template_arg_names
        )
        runtime_arg_names_str = ", ".join("unsigned int {}".format(x) for x in runtime_arg_names)
        code += "template <{}>\n".format(template_arg_names_str)
        code += "__inline_always void {name}({runtime_args})\n".format(
            name=sw_instruction.name, runtime_args=runtime_arg_names_str
        )
        code += "{\n"

        # Assert on registers used for 16/32 bit operations (they must be even), and swizzle input registers
        # This is not expressed in the csv so we must hardcoded it here
        if sw_instruction.name.endswith("_16"):
            args_even = ["Dest", "Src0", "Src1"]
            # Rotate and shift instructions which take the shift amount from a VE register, don't need that shift
            # register to be even (as it's only 8-bit)
            if re.match(r"(rol|[al]s[lr])r", hw_instruction.name, re.I):
                args_even.remove("Src1")
        elif sw_instruction.name.endswith("16_8"):
            args_even = ["Src0"]
        elif "swz_8" in sw_instruction.name:
            args_even = ["Src0", "Src1"]
        else:
            args_even = []

        # Assert on SW argument max values. We only need to do this for the template args.
        for arg_name in template_arg_names:
            if arg_name in ("Src0", "Src1", "Dest"):
                # Arguments specifying VE registers must be < 24 as there are only 24 registers in the VE.
                # This is not expressed in the csv file anywhere so must be hardcoded here.
                upper_bound = 24
            else:
                # Other arguments are limited by the number of bits
                upper_bound = 1 << sw_instruction.get_argument(arg_name).get_num_bits()
            code += '    static_assert({} < {}, "Argument out of range");\n'.format(
                arg_name, upper_bound
            )

            if arg_name in args_even:
                code += '    static_assert(({} % 2) == 0, "{}");\n'.format(
                    arg_name,
                    "Register number must be even for 16/32 bit values, and for swizzle inputs",
                )

        # The list of template args taken by the C++ intrinsics that we emit
        if hw_instruction.hw_instruction_type in ("MCR", "MCR2"):
            intrinsic_template_args = ["CPNUM", "op1", "op2", "CRn", "CRm"]
        elif hw_instruction.hw_instruction_type in ("MCRR", "MCRR2"):
            intrinsic_template_args = ["CPNUM", "op1", "CRm"]
        elif hw_instruction.hw_instruction_type in ("CDP", "CDP2"):
            intrinsic_template_args = ["CPNUM", "op1", "op2", "CRd", "CRn", "CRm"]
        else:
            assert False

        # Set local vars for each intrinsic template arg
        for intrinsic_template_arg in intrinsic_template_args:
            code += "    constexpr unsigned {} = {};\n".format(
                intrinsic_template_arg,
                self._get_cpp_argument(hw_instruction, sw_instruction, intrinsic_template_arg),
            )

        # Call the intrinsic with the template args and runtime args
        template_args_str = ", ".join(intrinsic_template_args)
        runtime_args_str = ", ".join(runtime_arg_names)
        code += "    {}<{}>({});\n".format(
            hw_instruction.hw_instruction_type.lower(),
            template_args_str,
            runtime_args_str,
        )

        # Nops
        code += "    constexpr unsigned wbCycle = VE_TIMING::{}::WRITE_BACK;\n".format(
            sw_instruction.name.upper()[3:]
        )
        code += "    nop<(COPRO_PIPELINE_DISABLE || (post_cc > wbCycle)) ? wbCycle : post_cc>();\n"

        code += "}\n"
        return code

    @staticmethod
    def _get_cpp_argument(hw_instruction, sw_instruction, hw_argument_name):
        argument = hw_instruction.get_argument_or_none(hw_argument_name)
        if argument is None:
            return "0 /* Not specified */"  # Not all arguments have a defined value, so we need a default
        bit_values = []
        # Note we loop in reverse to show MSBs first (the resulting code will be equivalent, but this is easier to read)
        for bit_idx in range(
            hw_instruction.get_argument(hw_argument_name).get_num_bits() - 1, -1, -1
        ):
            bit = hw_instruction.get_argument(hw_argument_name).bits[bit_idx]
            if bit is None:
                # Not all bits of every argument have a defined value, so we need a default
                bit_value = "0 /* Not specified */"
            elif bit.is_linked():
                bit_value = "GetBit({}, {})".format(bit.linked_argument_name, bit.linked_bit_idx)
            elif bit.is_constant():
                bit_value = str(bit.value)
            bit_values.append(bit_value)

        code = "Bits({})".format(", ".join(bit_values))
        return code

    def decode_hardware_instruction(self, hw_instruction_type, hw_args):
        """
        Given a description of an actual hardware instruction (with known argument values), e.g.
            hw_instruction_type = 'CDP'
            hw_args = { 'CPNUM': 4, 'op1': 5, 'op2': 0, 'CRd': 0, 'CRn': 0, 'CRm': 0 }
        Decodes this into the corresponding actual software instructon (with known argument values), e.g.
            ('ve_mov_8', { 'Dest': 16, 'Src0': 0 })
        """

        # Go through every instruction we know about, and see if any match
        for (hw_instruction, sw_instruction) in self._instructions:
            # Check it's the right type (CDP, MCR etc.)
            if hw_instruction.hw_instruction_type != hw_instruction_type:
                continue

            # Go through all the bits in the HW args, and make sure any fixed ones match
            match = True
            for hw_argument in hw_instruction.get_arguments():
                for i, bit in enumerate(hw_argument.bits):
                    if bit is not None and bit.is_constant():
                        expected_value = bit.value
                        actual_value = (hw_args[hw_argument.name] >> i) & 1
                        if expected_value != actual_value:
                            match = False

            if not match:
                continue

            # Now we know it's a match, figure out the values of the SW arguments, based on the values from the HW args
            sw_arg_values = {}
            for sw_argument in sw_instruction.get_arguments():
                sw_value = 0
                for i, bit in enumerate(sw_argument.bits):
                    if bit is not None and bit.is_linked():
                        bit_value = (hw_args[bit.linked_argument_name] >> bit.linked_bit_idx) & 1
                    else:
                        raise ValueError("All SW bits should be linked")
                    sw_value += bit_value << i
                sw_arg_values[sw_argument.name] = sw_value

            return (sw_instruction.name, sw_arg_values)

        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parses a .csv file of coprocessor instruction encodings (from the spec) "
        "into a database of instructions. This can be used in both directions - "
        "to export C++ function definitions which expose the SW interface to the HW instructions, "
        " and also to decode HW instructions into the more friendly SW description."
    )
    parser.add_argument(
        "-i",
        "--input_csv",
        help="Input path to csv file containing instruction encodings",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--input_descriptions",
        help="Input path to csv file containing instruction descriptions",
    )
    parser.add_argument("-o", "--output_header", help="Output path of generated C++ header file")
    parser.add_argument(
        "-t",
        "--input_timings",
        help="Input path to csv file containing instruction timings",
    )
    args = parser.parse_args()

    database = CoprocessorInstructionDatabase()
    database.load(args.input_csv, args.input_descriptions, args.input_timings)
    if args.output_header:
        database.export_cpp_functions(args.output_header)
