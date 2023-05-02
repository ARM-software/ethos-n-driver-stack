#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2021 Arm Limited.
# SPDX-License-Identifier: Apache-2.0
#

# Tests for CoprocessorInstructionDatabase.py

import unittest
import textwrap
import copy
from CoprocessorInstructionDatabase import (
    CoprocessorInstructionDatabase,
    Instruction,
    HwInstruction,
    Argument,
    ConstantBit,
    LinkedBit,
)


class TestCoprocessorInstructionDatabase(unittest.TestCase):
    # pylint: disable=protected-access,eval-used

    def test_parse_verilog_bitfield_part(self):
        d = CoprocessorInstructionDatabase()

        self.assertListEqual(
            d._parse_verilog_bitfield_part("5'b0101"),
            [
                ConstantBit(1),
                ConstantBit(0),
                ConstantBit(1),
                ConstantBit(0),
                ConstantBit(0),
            ],
        )
        self.assertListEqual(
            d._parse_verilog_bitfield_part("5'h9"),
            [
                ConstantBit(1),
                ConstantBit(0),
                ConstantBit(0),
                ConstantBit(1),
                ConstantBit(0),
            ],
        )
        self.assertListEqual(
            d._parse_verilog_bitfield_part("bob[2:1]"),
            [LinkedBit("bob", 1), LinkedBit("bob", 2)],
        )
        self.assertListEqual(d._parse_verilog_bitfield_part("bob[3]"), [LinkedBit("bob", 3)])

    def test_parse_verilog_bitfield(self):
        d = CoprocessorInstructionDatabase()

        self.assertListEqual(
            d._parse_verilog_bitfield("2'h1, bob[2:1], 5'hE"),
            [
                ConstantBit(0),
                ConstantBit(1),
                ConstantBit(1),
                ConstantBit(1),
                ConstantBit(0),
                LinkedBit("bob", 1),
                LinkedBit("bob", 2),
                ConstantBit(1),
                ConstantBit(0),
            ],
        )

    def test_parse_immediate_field(self):
        d = CoprocessorInstructionDatabase()

        self.assertEqual(
            d._parse_immediate_field("Imm5[4:0]=CPNUM[0], CRm[3:0]"),
            (
                "Imm5",
                [
                    LinkedBit("CRm", 0),
                    LinkedBit("CRm", 1),
                    LinkedBit("CRm", 2),
                    LinkedBit("CRm", 3),
                    LinkedBit("CPNUM", 0),
                ],
            ),
        )

    def test_process_row_simple(self):
        """
        Tests that parsing a simple instruction adds the expected information to the database.
        """
        database = CoprocessorInstructionDatabase()
        headings = [
            "Name",
            "COPRO_INST",
            "Opcode1",
            "Opcode2",
            "Src0",
            "Src1",
            "Dest",
            "Immediate",
        ]
        row = [
            "ADD.8",
            "CDP",
            "4'h0",
            "3'h0",
            "CPNUM[1], CRn[3:0]",
            "CPNUM[0], CRm[3:0]",
            "CPNUM[2],CRd[3:0]",
            "",
        ]
        database._process_row(row, headings)
        self.assertEqual(len(database._instructions), 1)
        (hw_instruction, sw_instruction) = database._instructions[0]

        self.assertEqual(hw_instruction._hw_instruction_type, "CDP")
        self.assertEqual(hw_instruction._name, "ADD.8")
        self.assertListEqual(
            list(hw_instruction.get_arguments()),
            [
                Argument(
                    "CPNUM",
                    [
                        LinkedBit("Src1", 4),
                        LinkedBit("Src0", 4),
                        LinkedBit("Dest", 4),
                    ],
                ),
                Argument(
                    "op1",
                    [ConstantBit(0), ConstantBit(0), ConstantBit(0), ConstantBit(0)],
                ),
                Argument("op2", [ConstantBit(0), ConstantBit(0), ConstantBit(0)]),
                Argument(
                    "CRd",
                    [
                        LinkedBit("Dest", 0),
                        LinkedBit("Dest", 1),
                        LinkedBit("Dest", 2),
                        LinkedBit("Dest", 3),
                    ],
                ),
                Argument(
                    "CRn",
                    [
                        LinkedBit("Src0", 0),
                        LinkedBit("Src0", 1),
                        LinkedBit("Src0", 2),
                        LinkedBit("Src0", 3),
                    ],
                ),
                Argument(
                    "CRm",
                    [
                        LinkedBit("Src1", 0),
                        LinkedBit("Src1", 1),
                        LinkedBit("Src1", 2),
                        LinkedBit("Src1", 3),
                    ],
                ),
            ],
        )

        self.assertEqual(sw_instruction._name, "ve_add_8")
        self.assertListEqual(
            list(sw_instruction.get_arguments()),
            [
                Argument(
                    "Dest",
                    [
                        LinkedBit("CRd", 0),
                        LinkedBit("CRd", 1),
                        LinkedBit("CRd", 2),
                        LinkedBit("CRd", 3),
                        LinkedBit("CPNUM", 2),
                    ],
                ),
                Argument(
                    "Src0",
                    [
                        LinkedBit("CRn", 0),
                        LinkedBit("CRn", 1),
                        LinkedBit("CRn", 2),
                        LinkedBit("CRn", 3),
                        LinkedBit("CPNUM", 1),
                    ],
                ),
                Argument(
                    "Src1",
                    [
                        LinkedBit("CRm", 0),
                        LinkedBit("CRm", 1),
                        LinkedBit("CRm", 2),
                        LinkedBit("CRm", 3),
                        LinkedBit("CPNUM", 0),
                    ],
                ),
            ],
        )

    def test_process_row_duplicate(self):
        """
        Tests that parsing a row which duplicates information works correctly. In this case SET_SWZSEL_REG_SEL defines
        twice the relationship between opcodes and the swzsel argument
        """
        database = CoprocessorInstructionDatabase()
        headings = [
            "Name",
            "COPRO_INST",
            "Extension bit",
            "Opcode1",
            "Opcode2",
            "Src0",
            "Src1",
            "Dest",
            "swzsel",
        ]
        row = [
            "SET_SWZSEL_REG_SEL",
            "MCR",
            "1'h0",
            "2'b1,swzsel[3]",
            "swzsel[2:0]",
            "",
            "",
            "",
            "op1[0],op2[2:0]",
        ]
        database._process_row(row, headings)
        self.assertEqual(len(database._instructions), 1)
        (hw_instruction, sw_instruction) = database._instructions[0]

        self.assertEqual(hw_instruction._hw_instruction_type, "MCR")
        self.assertEqual(hw_instruction._name, "SET_SWZSEL_REG_SEL")
        self.assertListEqual(
            list(hw_instruction.get_arguments()),
            [
                Argument("op1", [LinkedBit("swzsel", 3), ConstantBit(1), ConstantBit(0)]),
                Argument(
                    "op2",
                    [
                        LinkedBit("swzsel", 0),
                        LinkedBit("swzsel", 1),
                        LinkedBit("swzsel", 2),
                    ],
                ),
                Argument(
                    "Rt",
                    [
                        LinkedBit("Rt", 0),
                        LinkedBit("Rt", 1),
                        LinkedBit("Rt", 2),
                        LinkedBit("Rt", 3),
                        LinkedBit("Rt", 4),
                    ],
                ),
            ],
        )

        self.assertEqual(sw_instruction._name, "ve_set_swzsel_reg_sel")
        self.assertListEqual(
            list(sw_instruction.get_arguments()),
            [
                Argument(
                    "Rt",
                    [
                        LinkedBit("Rt", 0),
                        LinkedBit("Rt", 1),
                        LinkedBit("Rt", 2),
                        LinkedBit("Rt", 3),
                        LinkedBit("Rt", 4),
                    ],
                ),
                Argument(
                    "swzsel",
                    [
                        LinkedBit("op2", 0),
                        LinkedBit("op2", 1),
                        LinkedBit("op2", 2),
                        LinkedBit("op1", 0),
                    ],
                ),
            ],
        )

    def test_process_row_immediate(self):
        """
        Tests that parsing a row which has a value in the "Immediate" column.
        """
        database = CoprocessorInstructionDatabase()
        headings = [
            "Name",
            "COPRO_INST",
            "Opcode1",
            "Opcode2",
            "Src0",
            "Src1",
            "Dest",
            "Immediate",
        ]
        row = [
            "ASRSAT.16.8",
            "CDP2",
            "4'h8",
            "3'h0",
            "CPNUM[1], CRn[3:0]",
            "",
            "CPNUM[2],CRd[3:0]",
            "Imm5[4:0]=CPNUM[0], CRm[3:0]",
        ]
        database._process_row(row, headings)
        self.assertEqual(len(database._instructions), 1)
        (hw_instruction, sw_instruction) = database._instructions[0]

        self.assertEqual(hw_instruction._hw_instruction_type, "CDP2")
        self.assertEqual(hw_instruction._name, "ASRSAT.16.8")
        self.assertListEqual(
            list(hw_instruction.get_arguments()),
            [
                Argument(
                    "CPNUM",
                    [LinkedBit("Imm5", 4), LinkedBit("Src0", 4), LinkedBit("Dest", 4)],
                ),
                Argument(
                    "op1",
                    [ConstantBit(0), ConstantBit(0), ConstantBit(0), ConstantBit(1)],
                ),
                Argument("op2", [ConstantBit(0), ConstantBit(0), ConstantBit(0)]),
                Argument(
                    "CRd",
                    [
                        LinkedBit("Dest", 0),
                        LinkedBit("Dest", 1),
                        LinkedBit("Dest", 2),
                        LinkedBit("Dest", 3),
                    ],
                ),
                Argument(
                    "CRn",
                    [
                        LinkedBit("Src0", 0),
                        LinkedBit("Src0", 1),
                        LinkedBit("Src0", 2),
                        LinkedBit("Src0", 3),
                    ],
                ),
                Argument(
                    "CRm",
                    [
                        LinkedBit("Imm5", 0),
                        LinkedBit("Imm5", 1),
                        LinkedBit("Imm5", 2),
                        LinkedBit("Imm5", 3),
                    ],
                ),
            ],
        )

        self.assertEqual(sw_instruction._name, "ve_asrsat_16_8")
        self.assertListEqual(
            list(sw_instruction.get_arguments()),
            [
                Argument(
                    "Dest",
                    [
                        LinkedBit("CRd", 0),
                        LinkedBit("CRd", 1),
                        LinkedBit("CRd", 2),
                        LinkedBit("CRd", 3),
                        LinkedBit("CPNUM", 2),
                    ],
                ),
                Argument(
                    "Src0",
                    [
                        LinkedBit("CRn", 0),
                        LinkedBit("CRn", 1),
                        LinkedBit("CRn", 2),
                        LinkedBit("CRn", 3),
                        LinkedBit("CPNUM", 1),
                    ],
                ),
                Argument(
                    "Imm5",
                    [
                        LinkedBit("CRm", 0),
                        LinkedBit("CRm", 1),
                        LinkedBit("CRm", 2),
                        LinkedBit("CRm", 3),
                        LinkedBit("CPNUM", 0),
                    ],
                ),
            ],
        )

    def test_export_cpp_function(self):
        """
        Tests that converting a database entry to a CPP function produces the expected result.
        An imaginary instruction is added to the database, to exercise all the functionality we want to test.
        """
        hw_instruction = HwInstruction("IMAG.16", "MCR")
        hw_instruction.add_argument("CPNUM", [None, LinkedBit("swArg", 0), LinkedBit("Src0", 0)])
        hw_instruction.add_argument(
            "op1", [ConstantBit(1), LinkedBit("swArg", 1), LinkedBit("Src0", 1)]
        )
        hw_instruction.add_argument(
            "CRn", [ConstantBit(1), LinkedBit("swArg", 2), LinkedBit("Src0", 2)]
        )
        hw_instruction.add_argument(
            "Rt",
            [
                LinkedBit("Rt", 0),
                LinkedBit("Rt", 1),
                LinkedBit("Rt", 2),
                LinkedBit("Rt", 3),
                LinkedBit("Rt", 4),
            ],
        )

        sw_instruction = Instruction("ve_imag_16")
        sw_instruction.add_argument(
            "Src0", [LinkedBit("CPNUM", 0), LinkedBit("op1", 0), LinkedBit("CRn", 0)]
        )
        sw_instruction.add_argument(
            "swArg", [LinkedBit("CPNUM", 1), LinkedBit("op1", 1), LinkedBit("CRn", 1)]
        )
        sw_instruction.add_argument(
            "Rt",
            [
                LinkedBit("Rt", 0),
                LinkedBit("Rt", 1),
                LinkedBit("Rt", 2),
                LinkedBit("Rt", 3),
                LinkedBit("Rt", 4),
            ],
        )

        database = CoprocessorInstructionDatabase()
        database._descriptions = {"IMAG.16": ["This", "is", "description"]}

        self.assertEqual(
            database._export_cpp_function(hw_instruction, sw_instruction),
            textwrap.dedent(
                r"""
// This
// is
// description
template <unsigned int Src0, unsigned int swArg, unsigned int post_cc = 0>
__inline_always void ve_imag_16(unsigned int Rt)
{
    static_assert(Src0 < 24, "Argument out of range");
    static_assert((Src0 % 2) == 0, "Register number must be even for 16/32 bit values, and for swizzle inputs");
    static_assert(swArg < 8, "Argument out of range");
    constexpr unsigned CPNUM = Bits(GetBit(Src0, 0), GetBit(swArg, 0), 0 /* Not specified */);
    constexpr unsigned op1 = Bits(GetBit(Src0, 1), GetBit(swArg, 1), 1);
    constexpr unsigned op2 = 0 /* Not specified */;
    constexpr unsigned CRn = Bits(GetBit(Src0, 2), GetBit(swArg, 2), 1);
    constexpr unsigned CRm = 0 /* Not specified */;
    mcr<CPNUM, op1, op2, CRn, CRm>(Rt);
    constexpr unsigned wbCycle = VE_TIMING::IMAG_16::WRITE_BACK;
    nop<(COPRO_PIPELINE_DISABLE || (post_cc > wbCycle)) ? wbCycle : post_cc>();
}
            """
            )[1:],
        )

    def test_decode_hardware_instruction(self):
        """
        Tests that asking the database to decode a hardware instruction returns the correct instruction with the
        correct SW argument values
        """
        database = CoprocessorInstructionDatabase()

        # Define the imaginary instruction we'll be decoding
        hw_instruction = HwInstruction("IMAG.16", "MCR")
        hw_instruction.add_argument("CPNUM", [None, LinkedBit("swArg", 0), LinkedBit("Src0", 0)])
        hw_instruction.add_argument(
            "op1", [ConstantBit(1), LinkedBit("swArg", 1), LinkedBit("Src0", 1)]
        )
        hw_instruction.add_argument(
            "CRn", [ConstantBit(1), LinkedBit("swArg", 2), LinkedBit("Src0", 2)]
        )
        hw_instruction.add_argument(
            "Rt",
            [
                LinkedBit("Rt", 0),
                LinkedBit("Rt", 1),
                LinkedBit("Rt", 2),
                LinkedBit("Rt", 3),
                LinkedBit("Rt", 4),
            ],
        )

        sw_instruction = Instruction("ve_imag_16")
        sw_instruction.add_argument(
            "Src0", [LinkedBit("CPNUM", 0), LinkedBit("op1", 0), LinkedBit("CRn", 0)]
        )
        sw_instruction.add_argument(
            "swArg", [LinkedBit("CPNUM", 1), LinkedBit("op1", 1), LinkedBit("CRn", 1)]
        )
        sw_instruction.add_argument(
            "Rt",
            [
                LinkedBit("Rt", 0),
                LinkedBit("Rt", 1),
                LinkedBit("Rt", 2),
                LinkedBit("Rt", 3),
                LinkedBit("Rt", 4),
            ],
        )

        # Before we add the real one, add some decoys (i.e. other instructions that shouldn't match)
        hw_instruction_wrong_type, sw_instruction_wrong_type = copy.deepcopy(
            hw_instruction
        ), copy.deepcopy(sw_instruction)
        hw_instruction_wrong_type._hw_instruction_type = "CDP"
        hw_instruction_wrong_type._name = "WRONG"
        sw_instruction_wrong_type._name = "WRONG"
        database._instructions.append((hw_instruction_wrong_type, sw_instruction_wrong_type))

        hw_instruction_wrong_op1, sw_instruction_wrong_op1 = copy.deepcopy(
            hw_instruction
        ), copy.deepcopy(sw_instruction)
        hw_instruction_wrong_op1._arguments["op1"]._bits[0] = ConstantBit(0)
        hw_instruction_wrong_op1._name = "WRONG"
        sw_instruction_wrong_op1._name = "WRONG"
        database._instructions.append((hw_instruction_wrong_op1, sw_instruction_wrong_op1))

        # Add the real instruction at the end
        database._instructions.append((hw_instruction, sw_instruction))

        # Decode it
        self.assertEqual(
            database.decode_hardware_instruction("MCR", {"CPNUM": 6, "op1": 7, "CRn": 1, "Rt": 3}),
            ("ve_imag_16", {"Rt": 3, "Src0": 6, "swArg": 3}),
        )


if __name__ == "__main__":
    unittest.main()
