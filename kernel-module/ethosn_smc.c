/*
 *
 * (C) COPYRIGHT 2021 Arm Limited. All rights reserved.
 *
 * This program is free software and is provided to you under the terms of the
 * GNU General Public License version 2 as published by the Free Software
 * Foundation, and any use by you of this program is subject to the terms
 * of such GNU licence.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-2.0.html.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "ethosn_smc.h"

/* Compatible Arm Ethos-N NPU (NPU) SiP service version */
#define ETHOSN_SIP_MAJOR_VERSION        0
#define ETHOSN_SIP_MINOR_VERSION        1

/* SMC reset calls */
#define ETHOSN_SMC_VERSION              0xc2000050
#define ETHOSN_SMC_IS_SECURE            0xc2000051
#define ETHOSN_SMC_CORE_HARD_RESET      0xc2000052
#define ETHOSN_SMC_CORE_SOFT_RESET      0xc2000053

#define ETHOSN_SMC_CALL(cmd, res) \
	arm_smccc_smc(cmd, 0, 0, 0, 0, 0, 0, 0, res)

#define ETHOSN_SMC_CORE_CALL(cmd, core_id, res)	\
	arm_smccc_smc(cmd, core_id, 0, 0, 0, 0, 0, 0, res)

int ethosn_smc_version_check(struct ethosn_core *core)
{
	struct arm_smccc_res res = { 0 };

	ETHOSN_SMC_CALL(ETHOSN_SMC_VERSION, &res);
	if (res.a0 < 0) {
		dev_warn(core->dev,
			 "Arm Ethos-N NPU SiP service not available\n");

		return -EFAULT;
	}

	if (res.a0 != ETHOSN_SIP_MAJOR_VERSION ||
	    res.a1 < ETHOSN_SIP_MINOR_VERSION) {
		dev_warn(core->dev,
			 "Incompatible Arm Ethos-N NPU SiP service version.");

		return -EINVAL;
	}

	return 0;
}

int ethosn_smc_is_secure(struct ethosn_core *core)
{
	struct arm_smccc_res res = { 0 };

	ETHOSN_SMC_CALL(ETHOSN_SMC_IS_SECURE, &res);
	if (res.a0 < 0) {
		dev_err(core->dev,
			"Arm Ethos-N NPU SiP service not available");

		return -EFAULT;
	}

	return res.a0;
}

int ethosn_smc_core_reset(struct ethosn_core *core,
			  int hard_reset)
{
	struct arm_smccc_res res = { 0 };
	const u32 smc_reset_call = hard_reset ? ETHOSN_SMC_CORE_HARD_RESET :
				   ETHOSN_SMC_CORE_SOFT_RESET;

	ETHOSN_SMC_CORE_CALL(smc_reset_call, core->core_id, &res);
	if (res.a0) {
		dev_warn(core->dev, "Failed to %s reset the hardware. %ld\n",
			 hard_reset ? "hard" : "soft", res.a0);

		return -EFAULT;
	}

	return 0;
}
