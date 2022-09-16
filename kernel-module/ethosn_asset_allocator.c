/*
 *
 * (C) COPYRIGHT 2022 Arm Limited.
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
#include "ethosn_asset_allocator.h"

#include <linux/device.h>
#include <linux/module.h>
#include <linux/of_platform.h>
#include <linux/platform_device.h>

static int ethosn_asset_allocator_pdev_remove(struct platform_device *pdev)
{
	struct ethosn_device *ethosn = ethosn_driver(pdev);
	struct ethosn_dma_allocator *asset_allocator =
		dev_get_drvdata(&pdev->dev);
	uint32_t alloc_id = asset_allocator->alloc_id;
	int ret = 0;

	dev_info(&pdev->dev, "Removing asset allocator");

	if (!ethosn) {
		dev_err(&pdev->dev, "ethosn NULL");

		return -EINVAL;
	}

	if (!ethosn->asset_allocator[alloc_id]) {
		dev_err(&pdev->dev, "asset_allocator NULL");

		return -EINVAL;
	}

	of_platform_depopulate(&pdev->dev);

	ret = ethosn_dma_top_allocator_destroy(&pdev->dev,
					       &ethosn->
					       asset_allocator[alloc_id]);

	if (ret)
		return ret;

	dev_set_drvdata(&pdev->dev, NULL);

	return ret;
}

static int ethosn_asset_allocator_pdev_probe(struct platform_device *pdev)
{
	struct ethosn_device *ethosn = ethosn_driver(pdev);
	struct ethosn_dma_allocator *asset_allocator;

	int ret = 0;

	dev_info(&pdev->dev, "Probing asset allocator\n");

	if (IS_ERR_OR_NULL(ethosn)) {
		dev_err(&pdev->dev, "Invalid parent device driver");

		return -EINVAL;
	}

	asset_allocator =
		ethosn_dma_top_allocator_create(
			ethosn->dev,
			ETHOSN_ALLOCATOR_ASSET);

	if (IS_ERR_OR_NULL(asset_allocator))
		return -ENOMEM;

	asset_allocator->alloc_id = ethosn->num_asset_allocs;

	ethosn->asset_allocator[ethosn->num_asset_allocs] = asset_allocator;

	ethosn->num_asset_allocs++;

	dev_set_drvdata(&pdev->dev, asset_allocator);

	ret = of_platform_default_populate(pdev->dev.of_node, NULL, &pdev->dev);
	if (ret)
		dev_err(&pdev->dev, "Failed to populate child devices\n");

	return ret;
}

static const struct of_device_id ethosn_asset_allocator_child_pdev_match[] = {
	{ .compatible = ETHOSN_ASSET_ALLOC_DRIVER_NAME },
	{ /* Sentinel */ },
};

MODULE_DEVICE_TABLE(of, ethosn_asset_allocator_child_pdev_match);

static struct platform_driver ethosn_asset_allocator_pdev_driver = {
	.probe                  = &ethosn_asset_allocator_pdev_probe,
	.remove                 = &ethosn_asset_allocator_pdev_remove,
	.driver                 = {
		.name           = ETHOSN_ASSET_ALLOC_DRIVER_NAME,
		.owner          = THIS_MODULE,
		.of_match_table = of_match_ptr(
			ethosn_asset_allocator_child_pdev_match),
		.pm             = NULL,
	},
};

int ethosn_asset_allocator_platform_driver_register(void)
{
	pr_info("Registering %s", ETHOSN_ASSET_ALLOC_DRIVER_NAME);

	return platform_driver_register(&ethosn_asset_allocator_pdev_driver);
}

void ethosn_asset_allocator_platform_driver_unregister(void)
{
	pr_info("Unregistering %s", ETHOSN_ASSET_ALLOC_DRIVER_NAME);
	platform_driver_unregister(&ethosn_asset_allocator_pdev_driver);
}
