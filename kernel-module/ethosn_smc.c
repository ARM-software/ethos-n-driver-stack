/*
 *
 * (C) COPYRIGHT 2021 Arm Limited.
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

/* Compatible SiP service version */
#define ETHOSN_SIP_MAJOR_VERSION        1
#define ETHOSN_SIP_MINOR_VERSION        0

/* SMC reset calls */
#define ETHOSN_SMC_VERSION              0xc2000050
#define ETHOSN_SMC_IS_SECURE            0xc2000051
#define ETHOSN_SMC_CORE_HARD_RESET      0xc2000052
#define ETHOSN_SMC_CORE_SOFT_RESET      0xc2000053

static inline long __must_check ethosn_smc_core_call(u32 cmd,
						     phys_addr_t core_addr,
						     struct arm_smccc_res *res)
{
	arm_smccc_smc(cmd, core_addr, 0, 0, 0, 0, 0, 0, res);

	return (long)res->a0;
}

static inline long __must_check ethosn_smc_call(u32 cmd,
						struct arm_smccc_res *res)
{
	return ethosn_smc_core_call(cmd, 0U, res);
}

int ethosn_smc_version_check(const struct device *dev)
{
	struct arm_smccc_res res = { 0 };

	if (ethosn_smc_call(ETHOSN_SMC_VERSION, &res) < 0) {
		dev_warn(dev, "SiP service not available.\n");

		return -ENXIO;
	}

	if (res.a0 != ETHOSN_SIP_MAJOR_VERSION ||
	    res.a1 < ETHOSN_SIP_MINOR_VERSION) {
		dev_warn(dev, "Incompatible SiP service version.\n");

		return -EPROTO;
	}

	return 0;
}

int ethosn_smc_is_secure(const struct device *dev,
			 phys_addr_t core_addr)
{
	struct arm_smccc_res res = { 0 };

	if (ethosn_smc_core_call(ETHOSN_SMC_IS_SECURE, core_addr, &res) < 0) {
		dev_err(dev, "SiP service not available.\n");

		return -ENXIO;
	}

	if (res.a0 > 1U) {
		dev_err(dev, "Invalid device secure status.\n");

		return -EPROTO;
	}

	return res.a0;
}

int ethosn_smc_core_reset(const struct device *dev,
			  phys_addr_t core_addr,
			  int hard_reset)
{
	struct arm_smccc_res res = { 0 };
	const u32 smc_reset_call = hard_reset ? ETHOSN_SMC_CORE_HARD_RESET :
				   ETHOSN_SMC_CORE_SOFT_RESET;

	if (ethosn_smc_core_call(smc_reset_call, core_addr, &res)) {
		dev_warn(dev, "Failed to %s reset the hardware: %ld\n",
			 hard_reset ? "hard" : "soft", res.a0);

		return -EFAULT;
	}

	return 0;
}
