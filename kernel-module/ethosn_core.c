/*
 *
 * (C) COPYRIGHT 2020-2021 Arm Limited. All rights reserved.
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

#include "ethosn_device.h"
#include "ethosn_core.h"

#include <linux/kernel.h>
#include <linux/device.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/iommu.h>

#define ETHOSN_CORE_DRIVER_NAME    "ethosn-core"
#define ETHOSN_CORE_NUM_MAX        64

static struct ethosn_device *ethosn_driver(struct platform_device *pdev)
{
	struct ethosn_device *ethosn = dev_get_drvdata(pdev->dev.parent);

	return ethosn;
}

static int ethosn_child_pdev_remove(struct platform_device *pdev)
{
	dev_dbg(&pdev->dev, "Removed the ethosn-core device\n");

	return 0;
}

static int ethosn_child_pdev_probe(struct platform_device *pdev)
{
	struct ethosn_device *ethosn = ethosn_driver(pdev);
	int core_id;
	int core_count = of_get_child_count(pdev->dev.parent->of_node);
	int ret = 0;

	dev_info(&pdev->dev, "Probing core");

	if (IS_ERR_OR_NULL(ethosn)) {
		dev_err(&pdev->dev, "Invalid Ethosn parent device driver");

		return -EINVAL;
	}

	if (core_count > ETHOSN_CORE_NUM_MAX) {
		dev_err(&pdev->dev, "Invalid number of cores, max = %d\n",
			ETHOSN_CORE_NUM_MAX);

		return -EINVAL;
	}

	core_id = ethosn->num_cores;

	if (core_id >= core_count) {
		dev_err(&pdev->dev, "Invalid core id enumeration (%d)\n",
			core_id);

		return -EINVAL;
	}

	/* Allocating the core device (ie struct ethosn_core)
	 * Allocated against parent device
	 */
	ethosn->core[core_id] = devm_kzalloc(
		pdev->dev.parent,
		sizeof(struct ethosn_core),
		GFP_KERNEL);

	if (!ethosn->core[core_id])
		return -ENOMEM;

	/* Link child device object */
	ethosn->core[core_id]->dev = &pdev->dev;
	ethosn->core[core_id]->core_id = core_id;
	ethosn->core[core_id]->parent = ethosn;

	dev_dbg(&pdev->dev, "Ethosn-core probed\n");

	++ethosn->num_cores;

	return ret;
}

static const struct of_device_id ethosn_child_pdev_match[] = {
	{ .compatible = ETHOSN_CORE_DRIVER_NAME },
	{ /* Sentinel */ },
};

MODULE_DEVICE_TABLE(of, ethosn_child_pdev_match);

static struct platform_driver ethosn_child_pdev_driver = {
	.probe                  = &ethosn_child_pdev_probe,
	.remove                 = &ethosn_child_pdev_remove,
	.driver                 = {
		.name           = ETHOSN_CORE_DRIVER_NAME,
		.owner          = THIS_MODULE,
		.of_match_table = of_match_ptr(ethosn_child_pdev_match),
	},
};

int ethosn_core_platform_driver_register(void)
{
	pr_info("Registering %s", ETHOSN_CORE_DRIVER_NAME);

	return platform_driver_register(&ethosn_child_pdev_driver);
}

void ethosn_core_platform_driver_unregister(void)
{
	pr_info("Unregistering %s", ETHOSN_CORE_DRIVER_NAME);
	platform_driver_unregister(&ethosn_child_pdev_driver);
}
